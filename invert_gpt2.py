"""
Embedding Inversion Script: GPT-2 Generator

This script inverts text embeddings using GPT-2 as the language model for generation.
The encoder model can be specified via command line parameter.

Credit: the generator (learnable key-value offsets on dummy tokens, fluency loss,
sentence-level updates, decoding constraints) is adapted from the implementation of
"Zero-Shot Video Captioning by Evolving Pseudo-tokens" (Tewel et al., BMVC 2023),
https://github.com/YoadTew/zero-shot-video-to-text, with CLIP replaced by a text
embedding model scored against the target embedding.

Usage:
    python invert_gpt2.py --encoder-model sentence-transformers/gtr-t5-base --dataset msmarco --max-samples 100
    python invert_gpt2.py --encoder-model sentence-transformers/all-MiniLM-L6-v2 --dataset /path/to/custom.csv
"""

import argparse
import csv
import os
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)

from SentenceTransformerWrapper import SentenceTransformerWrapper


NOISE_BASE_SEED = 1_234_567


def _deterministic_noise(dim, sample_id, device, dtype):
    """Reproducible Gaussian noise vector of length `dim` for one sample.

    Seeded by NOISE_BASE_SEED + sample_id and drawn on CPU so the identical
    vector is reproduced in every pipeline phase (KV/HC/refine) regardless of
    which GPU runs it. sample_id=None falls back to a fresh (non-reproducible)
    draw. This is the invariant that lets the whole pipeline attack ONE fixed
    noisy target instead of a different random target per phase.
    """
    if sample_id is None:
        return torch.randn(dim, device=device, dtype=dtype)
    g = torch.Generator()
    g.manual_seed(NOISE_BASE_SEED + int(sample_id))
    return torch.randn(dim, generator=g).to(device=device, dtype=dtype)


class NoisySentenceTransformerWrapper(SentenceTransformerWrapper):
    """SentenceTransformerWrapper that can inject Gaussian noise into embeddings.

    Used to simulate a defense where the embedding provider adds noise to stored
    embeddings. Noise is only applied via get_logits_noisy(), not on every call —
    the attacker has clean query access to the encoder.
    """
    def __init__(self, *args, noise_level=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_level = noise_level

    def get_logits_noisy(self, texts, prompt_name=None, sample_id=None):
        """Encode with Gaussian noise added (for target embeddings only).

        Noise is deterministic in sample_id so the same noisy target is
        reproduced across pipeline phases (see _deterministic_noise).
        """
        embs = super().get_logits(texts, prompt_name=prompt_name)
        if self.noise_level > 0:
            # Unit-normalize the base BEFORE adding noise so the noisy target
            # matches HC/refine (which encode with normalize_embeddings=True).
            # Without this, the same noise vector added to a different-magnitude
            # base yields a different normalized target across phases.
            embs = F.normalize(embs, p=2, dim=-1)
            noise = _deterministic_noise(embs.shape[-1], sample_id, embs.device, embs.dtype)
            embs = embs + noise * self.noise_level
            embs = F.normalize(embs, p=2, dim=-1)
        return embs


# =============================================================================
# CONFIGURATION (Generator is hardcoded, encoder is parameterized)
# =============================================================================

DEFAULT_ENCODER = "sentence-transformers/gtr-t5-base"
HARDCODED_GENERATOR = "gpt2-medium"


# =============================================================================
# GENERATOR CLASS
# =============================================================================

class Generator:
    def __init__(
            self,
            seed=42,
            generator_model=HARDCODED_GENERATOR,
            encoder_instance=None,
            use_query_prompt=False,
            target_seq_max_length=30,
            num_dummy_tokens=15,
            sentence_iterations=150,
            sampling_top_k=50,
            learning_rate=0.05,
            cls_loss_scale=1.0,
            fluency_loss_scale=1.0,
            weight_decay_scale=0.3,
            repetition_penalty=5.0,
            dummy_starter=None,
            similarity_temperature=0.008,
            loss_aggregation="mean",
            entity_penalty=50,
            short_token_penalty=30,
            min_token_length=2,
            num_loss_computations=3,
            kv_layers=None,
            boost_dedup=True,
    ):
        if encoder_instance is None:
            raise ValueError("An encoder instance must be provided")

        self.encoder = encoder_instance
        self.use_query_prompt = use_query_prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = seed
        self.reset_random_seed()

        # Initialize language model
        self.lm_tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            generator_model,
            output_hidden_states=True,
            torch_dtype=torch.float32,
        )
        self.generator_model.to(self.device).eval()
        for param in self.generator_model.parameters():
            param.requires_grad = False

        if self.lm_tokenizer.pad_token is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token

        # Determine start token
        if self.lm_tokenizer.bos_token_id is not None:
            self.start_token_id = self.lm_tokenizer.bos_token_id
        else:
            self.start_token_id = self.lm_tokenizer.eos_token_id

        # KV cache initialization
        self.num_dummy_tokens = num_dummy_tokens
        if dummy_starter:
            starter_token_ids = self.lm_tokenizer.encode(dummy_starter, add_special_tokens=False)
            if len(starter_token_ids) < num_dummy_tokens:
                padding_needed = num_dummy_tokens - len(starter_token_ids)
                dummy_token_ids = [self.start_token_id] * padding_needed + starter_token_ids
            elif len(starter_token_ids) > num_dummy_tokens:
                dummy_token_ids = starter_token_ids[-num_dummy_tokens:]
            else:
                dummy_token_ids = starter_token_ids
        else:
            dummy_token_ids = [self.start_token_id] * num_dummy_tokens

        dummy_tokens = torch.tensor([dummy_token_ids], device=self.device, dtype=torch.long)

        with torch.no_grad():
            self.dummy_context_init = self.generator_model(dummy_tokens)["past_key_values"]

        num_layers = len(self.dummy_context_init)
        # Determine which layers get learnable offsets
        if kv_layers is None:
            self.trainable_layers = set(range(num_layers))  # all layers
        else:
            self.trainable_layers = set(kv_layers)
        frozen_count = num_layers - len(self.trainable_layers)
        if frozen_count > 0:
            print(f"  KV layers: training {len(self.trainable_layers)}/{num_layers} "
                  f"(frozen: {sorted(set(range(num_layers)) - self.trainable_layers)})")

        # Learnable offsets (only for trainable layers; frozen layers keep zero offsets)
        self.dummy_context_offset = [
            (
                torch.nn.Parameter(torch.zeros(k.shape).type_as(k)),
                torch.nn.Parameter(torch.zeros(v.shape).type_as(v)),
            )
            if layer_idx in self.trainable_layers
            else (
                torch.nn.Parameter(torch.zeros(k.shape).type_as(k), requires_grad=False),
                torch.nn.Parameter(torch.zeros(v.shape).type_as(v), requires_grad=False),
            )
            for layer_idx, (k, v) in enumerate(self.dummy_context_init)
        ]

        optimizer_params = [
            x for layer_idx, pair in enumerate(self.dummy_context_offset)
            for x in pair if layer_idx in self.trainable_layers
        ]
        self.dummy_optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=learning_rate,
            weight_decay=learning_rate * weight_decay_scale,
        )

        # End token (used for early stopping)
        end_token_ids = self.lm_tokenizer.encode(".", add_special_tokens=False)
        self.end_token = end_token_ids[0] if end_token_ids else self.lm_tokenizer.eos_token_id

        # Repetition penalties
        self.deter_small_repeat = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
        self.prevent_large_repeat = NoRepeatNGramLogitsProcessor(ngram_size=2)

        # Prevent early completion of generation
        self.prevent_early_finish = MinLengthLogitsProcessor(
            min_length=target_seq_max_length // 2,
            eos_token_id=self.end_token
        )

        # Token mappings and penalties
        self.token_to_similar_indices = self._build_similar_token_mapping()
        self.first_token_offset, self.other_token_offset = self._build_token_specific_penalties(entity_penalty)
        self.short_token_penalty = self._build_short_token_penalty(short_token_penalty, min_token_length)

        # Boost-dedup: prevent re-boosting tokens whose case/whitespace variants
        # have already been used in the current generation.
        self.boost_dedup = boost_dedup

        # Token boost: set externally by main() via generator.token_boost = ...
        self.token_boost = None

        # Best runs tracking
        self.best_runs = [{"score": 0, "reconstruction": None, "kv_state": None} for _ in range(10)]

        # Per-sample encoder embedding cache: {sentence: embedding_tensor}
        # Populated during get_similarity_loss and reused across iterations to
        # avoid re-encoding candidates the beam has already seen. Cleared when
        # a new Generator is created (one per sample).
        self._emb_cache = {}
        self._emb_cache_hits = 0
        self._emb_cache_misses = 0

        # Store params
        self.target_seq_max_length = target_seq_max_length
        self.sentence_iterations = sentence_iterations
        self.sampling_top_k = sampling_top_k
        self.cls_loss_scale = cls_loss_scale
        self.fluency_loss_scale = fluency_loss_scale
        self.similarity_temperature = similarity_temperature
        self.loss_aggregation = loss_aggregation
        self.num_loss_computations = num_loss_computations

    def reset_random_seed(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _build_similar_token_mapping(self, max_similar=8):
        """Group tokens by normalized form (case/space variants)."""
        vocab_size = self.generator_model.config.vocab_size
        # Keep on CPU for use with CPU logits during sampling
        mapping = torch.zeros((vocab_size, max_similar), dtype=torch.long)

        vocab = self.lm_tokenizer.get_vocab()
        normalized_to_ids = {}

        for token_str, token_id in vocab.items():
            if token_id >= vocab_size:
                continue
            try:
                decoded = self.lm_tokenizer.decode([token_id])
                normalized = decoded.strip().upper()
                if normalized:
                    if normalized not in normalized_to_ids:
                        normalized_to_ids[normalized] = []
                    normalized_to_ids[normalized].append(token_id)
            except:
                pass

        for normalized, ids in normalized_to_ids.items():
            for token_id in ids:
                for i, similar_id in enumerate(ids[:max_similar]):
                    mapping[token_id, i] = similar_id

        return mapping

    def _build_token_specific_penalties(self, entity_penalty=50):
        """Build separate penalty tensors for first token vs later tokens."""
        vocab_size = self.generator_model.config.vocab_size

        lower_chars = [chr(c) for c in range(ord("a"), ord("z") + 1)]
        upper_chars = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        special_chars = [" ", ",", "'"]

        unwanted_first_tokens = []
        unwanted_later_tokens = []

        vocab = self.lm_tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            if token_id >= vocab_size:
                continue
            try:
                decoded = self.lm_tokenizer.decode([token_id])
                if not set(decoded).issubset(set(lower_chars + upper_chars)):
                    unwanted_first_tokens.append(token_id)
                if not set(decoded).issubset(set(lower_chars + upper_chars + special_chars)):
                    if token_id != self.lm_tokenizer.eos_token_id:
                        unwanted_later_tokens.append(token_id)
            except:
                pass

        # Keep on CPU for use with CPU logits during sampling
        first_token_offset = torch.zeros((vocab_size,))
        first_token_offset[unwanted_first_tokens] = -entity_penalty
        other_token_offset = torch.zeros((vocab_size,))
        other_token_offset[unwanted_later_tokens] = -entity_penalty

        return first_token_offset, other_token_offset

    def _build_short_token_penalty(self, penalty=30, min_length=2):
        """Build penalty for tokens that decode to very short strings."""
        vocab_size = self.generator_model.config.vocab_size
        # Keep on CPU for use with CPU logits during sampling
        penalties = torch.zeros((vocab_size,))

        vocab = self.lm_tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            if token_id >= vocab_size:
                continue
            try:
                decoded = self.lm_tokenizer.decode([token_id])
                stripped = decoded.strip()
                if len(stripped) < min_length and len(stripped) > 0:
                    penalties[token_id] = -penalty
            except:
                pass

        for special_id in [
            self.lm_tokenizer.eos_token_id,
            self.lm_tokenizer.bos_token_id,
            self.lm_tokenizer.pad_token_id,
            self.end_token,
        ]:
            if special_id is not None and special_id < vocab_size:
                penalties[special_id] = 0.0

        return penalties

    def tinker_logits(self, logits, generated_tokens, first_token=False):
        """
        Apply constraints and penalties to raw logits before sampling.

        Args:
            logits: Model output logits [batch_size, vocab_size] (CPU tensor)
            generated_tokens: Tokens generated so far [batch_size, seq_len] (CPU tensor)
            first_token: If True, apply stricter filtering for the first token

        Returns:
            Adjusted logits with penalties applied
        """
        # 1. Small repetition penalty with similar tokens
        history_len = min(16, generated_tokens.size(1))
        history_tokens = generated_tokens[:, -history_len:]
        # Expand history tokens to include "similar" variants
        history_similar = self.token_to_similar_indices[history_tokens]
        history_similar = history_similar.view(history_tokens.size(0), -1)
        logits = self.deter_small_repeat(history_similar, logits)

        # 2. Exact n-gram repetition penalty
        logits = self.prevent_large_repeat(generated_tokens, logits)

        # 3. Prevent premature EOS (using MinLengthLogitsProcessor)
        logits = self.prevent_early_finish(generated_tokens, logits)

        # 4. Penalize disallowed tokens based on position
        if first_token:
            logits = logits + self.first_token_offset.unsqueeze(0)
        else:
            logits = logits + self.other_token_offset.unsqueeze(0)

        # 5. Short token penalty
        logits = logits + self.short_token_penalty.unsqueeze(0)

        # 6. Token boost (names, terms from embedding similarity)
        #    Dedup: once a boosted token (or any case/whitespace variant) has
        #    been used, zero its boost for the rest of the generation so we
        #    don't dilute the embedding with redundant tokens.
        if hasattr(self, 'token_boost') and self.token_boost is not None:
            if self.boost_dedup and generated_tokens.size(1) > 0:
                used = set()
                for t in generated_tokens[0].tolist():
                    used.update(self.token_to_similar_indices[t].tolist())
                used.discard(0)  # slot-0 sentinel for tokens without variants
                if used:
                    boost = self.token_boost.clone()
                    boost[torch.tensor(list(used), dtype=torch.long)] = 0
                    logits = logits + boost.unsqueeze(0)
                else:
                    logits = logits + self.token_boost.unsqueeze(0)
            else:
                logits = logits + self.token_boost.unsqueeze(0)

        # 7. Numerical stability guard
        logits = logits.masked_fill(torch.isnan(logits), float("-inf"))

        return logits

    @property
    def dummy_context(self):
        cache = DynamicCache()
        return cache.from_legacy_cache([
            (k + k_delta, v + v_delta)
            for (k, v), (k_delta, v_delta) in zip(
                self.dummy_context_init, self.dummy_context_offset
            )
        ])

    def update_best_runs(self, new_score, text, max_runs=10):
        kv_state = {
            "offsets": [(k.data.clone(), v.data.clone()) for k, v in self.dummy_context_offset],
        }
        new_run = {"score": new_score, "reconstruction": text, "kv_state": kv_state}
        self.best_runs.append(new_run)
        self.best_runs.sort(key=lambda x: x["score"], reverse=True)
        if len(self.best_runs) > max_runs:
            self.best_runs = self.best_runs[:max_runs]

    def load_kv_state(self, kv_state):
        current_state = {
            "offsets": [(k.data.clone(), v.data.clone()) for k, v in self.dummy_context_offset],
        }
        for i, (k_data, v_data) in enumerate(kv_state["offsets"]):
            self.dummy_context_offset[i][0].data.copy_(k_data)
            self.dummy_context_offset[i][1].data.copy_(v_data)
        return current_state

    def restore_kv_state(self, saved_state):
        for i, (k_data, v_data) in enumerate(saved_state["offsets"]):
            self.dummy_context_offset[i][0].data.copy_(k_data)
            self.dummy_context_offset[i][1].data.copy_(v_data)

    def generate(self, target_embedding):
        self.reset_random_seed()
        context = torch.tensor([[self.start_token_id]])

        for i in range(self.sentence_iterations):
            print(f"Iteration: {i}")
            generated_tokens, similarity_score = self.sample_with_context(target_embedding, context)
            generated_text = self.lm_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            self.update_best_runs(similarity_score, generated_text[0])

        total = self._emb_cache_hits + self._emb_cache_misses
        if total > 0:
            hit_rate = 100.0 * self._emb_cache_hits / total
            print(f"\n[Encoder cache] hits={self._emb_cache_hits}  misses={self._emb_cache_misses}  "
                  f"hit_rate={hit_rate:.1f}%  unique_encoded={len(self._emb_cache)}")

        print("\n=== Best Runs (sorted by similarity) ===")
        for i, run in enumerate(self.best_runs):
            if run["reconstruction"]:
                score_val = run["score"].item() if hasattr(run["score"], "item") else run["score"]
                print(f"\nRank {i + 1} | Similarity: {score_val:.4f}")
                print(f"Text: {run['reconstruction']}")
        return self.best_runs

    def sample_next_token(self, logits, generated_tokens, sampling_top_k=5, first_token=False):
        """
        Sample the next token from logits after applying constraints.

        Args:
            logits: Raw model logits [batch_size, vocab_size]
            generated_tokens: Tokens generated so far [batch_size, seq_len]
            sampling_top_k: Number of top tokens to sample from
            first_token: Whether this is the first generated token

        Returns:
            Next token ID [batch_size, 1]
        """
        # Apply all constraints and penalties
        logits = self.tinker_logits(logits, generated_tokens, first_token)

        # Narrow down to top k options
        option_logits, next_token_options = logits.topk(sampling_top_k, dim=-1)

        # Convert to probabilities and sample
        top_probs = F.softmax(option_logits, dim=-1) + 1e-10
        next_token_index = torch.multinomial(top_probs, num_samples=1)[0]
        next_token_id = next_token_options[:, next_token_index]

        return next_token_id

    def _get_encoder_embedding(self, text):
        """Get embedding from encoder, using query prompt if configured."""
        if self.use_query_prompt:
            return self.encoder.get_logits(text, prompt_name="query")
        return self.encoder.get_logits(text)

    def sample_with_context(self, target_embedding, context):
        """
        Generate tokens using only the dummy (learnable) KV cache.

        At each step, the full generated sequence is passed to the model along with
        the dummy context as past_key_values. No caching of generated tokens' KV values.
        """
        input_ids = context.to(self.device)
        generated_tokens = input_ids
        fluency_losses = []
        similarity_losses = []

        # Pre-compute steps at which to compute loss (evenly spaced)
        # For num_loss_computations=4 and max_len=30: steps {8, 15, 23, 30}
        loss_steps = set(
            round(self.target_seq_max_length * k / self.num_loss_computations)
            for k in range(1, self.num_loss_computations + 1)
        )

        for i in range(1, self.target_seq_max_length + 1):
            # Always pass full sequence with only the dummy cache as past_key_values
            shifted_outputs = self.generator_model(
                generated_tokens, past_key_values=self.dummy_context
            )
            logits = shifted_outputs["logits"][:, -1, :]

            # Compute fluency & similarity loss at evenly spaced intervals
            if i in loss_steps:
                # Get unshifted (no prefix) model output for fluency comparison
                unshifted_outputs = self.generator_model(generated_tokens)
                unshifted_logits = unshifted_outputs["logits"][:, -1, :]
                unshifted_probs = F.softmax(unshifted_logits, dim=-1) + 1e-10

                probs = F.softmax(logits, dim=-1) + 1e-10
                fluency_loss = -(unshifted_probs * probs.log()).sum()
                fluency_losses.append(fluency_loss)

                # Similarity loss
                cls_loss = self.get_similarity_loss(target_embedding, logits, generated_tokens)
                similarity_losses.append(cls_loss)

            # Sample next token (apply constraints on CPU for consistency)
            generated_tokens_cpu = generated_tokens.detach().cpu()
            logits_cpu = logits.detach().cpu()
            next_token = self.sample_next_token(
                logits_cpu, generated_tokens_cpu,
                sampling_top_k=self.sampling_top_k,
                first_token=(i == 1)
            )

            generated_tokens = torch.cat((generated_tokens, next_token.to(self.device)), dim=1)
            if next_token.item() == self.end_token:
                break

        # Loss aggregation
        n_loss_steps = len(fluency_losses) if fluency_losses else 1
        if self.loss_aggregation == "sum":
            final_fluency_loss = sum(fluency_losses) if fluency_losses else torch.tensor(0.0, device=self.device)
            final_similarity_loss = sum(similarity_losses) if similarity_losses else torch.tensor(0.0, device=self.device)
        else:
            final_fluency_loss = sum(fluency_losses) / n_loss_steps if fluency_losses else torch.tensor(0.0, device=self.device)
            final_similarity_loss = sum(similarity_losses) / n_loss_steps if similarity_losses else torch.tensor(0.0, device=self.device)

        total_loss = self.fluency_loss_scale * final_fluency_loss + self.cls_loss_scale * final_similarity_loss

        # Final similarity (use cache: this text will often reappear as a
        # candidate in later iterations' top-k)
        text = self.lm_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        final_text = text[0]
        if final_text in self._emb_cache:
            representation = self._emb_cache[final_text].to(target_embedding.device)
            self._emb_cache_hits += 1
        else:
            representation = self._get_encoder_embedding(final_text)
            if not isinstance(representation, torch.Tensor):
                representation = torch.tensor(representation, device=target_embedding.device)
            self._emb_cache[final_text] = representation.detach().cpu()
            self._emb_cache_misses += 1
        final_similarity = F.cosine_similarity(target_embedding, representation.unsqueeze(0), dim=1)

        flu_val = final_fluency_loss.item() if hasattr(final_fluency_loss, "item") else final_fluency_loss
        sim_val = final_similarity_loss.item() if hasattr(final_similarity_loss, "item") else final_similarity_loss
        loss_val = total_loss.item() if hasattr(total_loss, "item") else total_loss
        print(f"Loss: {loss_val:.4f} (flu: {flu_val:.2f}, sim: {sim_val:.2f}) | Similarity: {final_similarity.item():.4f} | {text}")

        # Backpropagation
        total_loss.backward()
        self.dummy_optimizer.step()
        self.dummy_optimizer.zero_grad()

        return generated_tokens, final_similarity

    def _batch_encode_cached(self, texts):
        """Batch-encode a list of sentences using the per-sample cache.
        Returns a CPU tensor [N, D] aligned with the input list order.
        """
        missing = [s for s in texts if s not in self._emb_cache]
        seen = set()
        to_encode = [s for s in missing if not (s in seen or seen.add(s))]
        if to_encode:
            with torch.no_grad():
                new_embs = self._get_encoder_embedding(to_encode)
            if not isinstance(new_embs, torch.Tensor):
                new_embs = torch.tensor(new_embs)
            new_embs = new_embs.detach().cpu()
            for j, s in enumerate(to_encode):
                self._emb_cache[s] = new_embs[j]
        self._emb_cache_hits += len(texts) - len(missing)
        self._emb_cache_misses += len(to_encode)
        return torch.stack([self._emb_cache[s] for s in texts])

    def get_similarity_loss(self, target_embedding, gen_logits, tokens_so_far):
        """
        KL divergence between model distribution and similarity-based target.

        For each of the top-k candidate tokens, computes the similarity of the
        resulting sentence to the target embedding and creates a target distribution.
        """
        # Apply constraints on CPU to get top-k candidates
        logits_copy = gen_logits.clone()
        logits_copy = self.tinker_logits(logits_copy.cpu(), tokens_so_far.cpu(), False)
        _, top_next_tokens = logits_copy.topk(self.sampling_top_k, -1)

        next_sentences = {}
        for token in top_next_tokens[0]:
            tokenized = torch.cat((tokens_so_far, token.unsqueeze(0).unsqueeze(0).to(self.device)), dim=1)
            sentence = self.lm_tokenizer.decode(tokenized[0], skip_special_tokens=True)
            next_sentences[token.item()] = sentence

        sentences_batch = list(next_sentences.values())
        token_list = list(next_sentences.keys())

        # Use per-sample encoder cache via the shared helper
        cls_representations = self._batch_encode_cached(sentences_batch)

        # Compute cosine similarity for ground truth distribution
        input_expanded = target_embedding.detach().cpu()[0].unsqueeze(0)
        ground_truth_logits = F.cosine_similarity(input_expanded, cls_representations, dim=1)

        # Create target distribution from similarities
        ground_truth_distr = F.softmax(ground_truth_logits / self.similarity_temperature, dim=0)

        # Get model's distribution over the candidate tokens
        gen_needed_logits = gen_logits[0][token_list].cpu()
        gen_needed_distr = F.softmax(gen_needed_logits, dim=0)

        return F.kl_div(gen_needed_distr.log(), ground_truth_distr, reduction="sum")

    def refine_with_learned_context(self, target_embedding, beam_size=30, top_k=30, max_tokens=30):
        """
        Beam search refinement using learned KV caches.

        Uses the learned dummy context (KV cache) to guide generation,
        scoring candidates by embedding similarity to target.
        """
        with torch.no_grad():
            initial_ids = torch.tensor([[self.start_token_id]], device=self.device)
            beams = [(initial_ids, 0.0)]

            for step in range(max_tokens):
                candidates = []

                for beam_ids, _ in beams:
                    # Use learned KV cache
                    logits = self.generator_model(beam_ids, past_key_values=self.dummy_context).logits[:, -1, :]
                    # Apply constraints on CPU
                    logits_cpu = self.tinker_logits(logits.cpu(), beam_ids.cpu())
                    _, top_tokens = logits_cpu.topk(top_k, dim=-1)

                    for token in top_tokens[0]:
                        new_ids = torch.cat([beam_ids, token.unsqueeze(0).unsqueeze(0).to(self.device)], dim=1)
                        candidates.append(new_ids)

                # Score all candidates by embedding similarity (with cache)
                texts = [self.lm_tokenizer.decode(ids[0], skip_special_tokens=True) for ids in candidates]
                missing = [t for t in texts if t not in self._emb_cache]
                seen = set()
                to_encode = [t for t in missing if not (t in seen or seen.add(t))]
                if to_encode:
                    new_embs = self._get_encoder_embedding(to_encode)
                    if not isinstance(new_embs, torch.Tensor):
                        new_embs = torch.tensor(new_embs)
                    new_embs = new_embs.detach().cpu()
                    for j, t in enumerate(to_encode):
                        self._emb_cache[t] = new_embs[j]
                self._emb_cache_hits += len(texts) - len(missing)
                self._emb_cache_misses += len(to_encode)

                scored = []
                target_cpu = target_embedding.cpu()
                for i, ids in enumerate(candidates):
                    emb = self._emb_cache[texts[i]]
                    sim = F.cosine_similarity(emb.unsqueeze(0), target_cpu).item()
                    scored.append((ids, sim))

                # Keep top beams
                scored.sort(key=lambda x: x[1], reverse=True)
                beams = scored[:beam_size]

                # Early stop if best ends with punctuation
                best_text = self.lm_tokenizer.decode(beams[0][0][0], skip_special_tokens=True)
                if best_text.endswith((".", "!", "?")):
                    break

            best_ids, best_score = beams[0]
            best_text = self.lm_tokenizer.decode(best_ids[0], skip_special_tokens=True)
            return best_text, best_score


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_texts_from_csv(csv_path, text_column="text", limit=None):
    """Load texts from a CSV file."""
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")
    texts = df[text_column].dropna().tolist()
    if limit:
        texts = texts[:limit]
    return texts


def get_dataset_path(dataset_name):
    """Get the path to a dataset CSV file."""
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / "dataset_csvs"

    dataset_map = {
        "msmarco": "msmarco_500_32tok.csv",
        "enron": "enron_aeslc_500_32tok.csv",
        "mimic": "MIMIC_III_420_32tok.csv",
        "mimic-iv": "mimic_IV_256tok.csv",
        "pubmed": "pubmed_500_32tok.csv",
    }

    if dataset_name in dataset_map:
        return str(dataset_dir / dataset_map[dataset_name])
    elif os.path.isfile(dataset_name):
        return dataset_name
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_map.keys())}")


def get_encoder_short_name(encoder_model):
    """Extract a short name from encoder model path for file naming."""
    name = encoder_model.split("/")[-1]
    # Remove common prefixes/suffixes for cleaner names
    for prefix in ["sentence-transformers-", "sentence-transformers_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Embedding Inversion with GPT-2 Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Encoder model (parameterized)
    parser.add_argument("--encoder-model", type=str, default=DEFAULT_ENCODER,
                        help="Encoder model (sentence-transformers compatible)")
    parser.add_argument("--use-query-prompt", action="store_true",
                        help="Use 'query' prompt for instruct-style encoders")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code when loading encoder model")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path to CSV")
    parser.add_argument("--text-column", type=str, default="text", help="Text column name in CSV")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--skip-samples", type=int, default=0, help="Skip first N samples")

    # Output
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")

    # Phase 1
    parser.add_argument("--sentence-iterations", type=int, default=150, help="KV optimization iterations")
    parser.add_argument("--max-sentence-length", type=int, default=32,
                        help="Max number of tokens to generate per reconstruction (Phase 1)")
    parser.add_argument("--max-length-mode", type=str, default="fixed", choices=["fixed", "auto"],
                        help="'fixed' uses --max-sentence-length for every sample; 'auto' sets a per-sample "
                             "length from the source's generator-token count (round(1.15x), clamped to [16, cap])")
    parser.add_argument("--max-length-cap", type=int, default=96,
                        help="Upper bound on the per-sample reconstruction length in 'auto' mode")
    parser.add_argument("--num-dummy-tokens", type=int, default=15, help="Number of prefix tokens")
    parser.add_argument("--learning-rate", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--sampling-top-k", type=int, default=50, help="Top-k sampling")

    # Loss
    parser.add_argument("--cls-loss-scale", type=float, default=1.0, help="Similarity loss scale")
    parser.add_argument("--fluency-loss-scale", type=float, default=1.0, help="Fluency loss scale")
    parser.add_argument("--loss-aggregation", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--similarity-temperature", type=float, default=0.008, help="Similarity softmax temp")
    parser.add_argument("--num-loss-computations", type=int, default=3,
                        help="Number of loss computations per generation (evenly spaced)")

    # KV layer selection
    parser.add_argument("--kv-layers", type=str, default=None,
                        help="Which layers to train KV offsets for. "
                             "Format: '6-20' (range), '8,10,12,14' (list), or 'middle' (auto). "
                             "Default: all layers. GPT-2 medium has 24 layers (0-23).")

    # Boost dedup (case/whitespace variants via token_to_similar_indices)
    parser.add_argument("--no-boost-dedup", action="store_true",
                        help="Disable dedup of boosted tokens. "
                             "By default, once 'CPR' is used, ' CPR'/'cpr'/'CPR' lose boost.")

    # Penalties
    parser.add_argument("--entity-penalty", type=float, default=5, help="Entity token penalty (low to allow names)")
    parser.add_argument("--short-token-penalty", type=float, default=30, help="Short token penalty")
    parser.add_argument("--min-token-length", type=int, default=2, help="Min token length")

    # Token boost (from knowledge bank)
    parser.add_argument("--boost-strength", type=float, default=3.0,
                        help="Logit boost strength (0=disable)")
    parser.add_argument("--num-boost-tokens", type=int, default=20,
                        help="Top-K matches per attribute to boost")
    parser.add_argument("--boost-attributes", nargs="+",
                        default=["vocab"],
                        help="'vocab' for direct vocab boost (default), or knowledge bank attributes like first_names last_names diagnoses")
    parser.add_argument("--capitalize-boost", type=float, default=1.0,
                        help="Extra multiplier for capitalized tokens (1.0=disable)")
    parser.add_argument("--boost-min-token-length", type=int, default=2,
                        help="Minimum decoded token length (in chars) to be eligible for vocab boost. "
                             "Higher values exclude prepositions/short words like 'he','on','it'.")
    parser.add_argument("--boost-template", type=str, default=None,
                        help="Template for boost embeddings: None=per-attribute default, 'mask', 'bare', or custom")
    parser.add_argument("--vocab-boost-mask", action="store_true", default=True,
                        help="Use [MASK] prefix for vocab boost embeddings (default: True)")
    parser.add_argument("--no-vocab-boost-mask", action="store_false", dest="vocab_boost_mask",
                        help="Disable [MASK] prefix for vocab boost embeddings")

    # Phase 2
    parser.add_argument("--enable-beam-search", action="store_true",
                        help="Enable Phase 2 beam search refinement (disabled by default)")
    parser.add_argument("--refine-top-n", type=int, default=5, help="Top N to refine")
    parser.add_argument("--beam-size", type=int, default=30, help="Beam size")
    parser.add_argument("--beam-top-k", type=int, default=30, help="Beam top-k")
    parser.add_argument("--beam-max-tokens", type=int, default=35, help="Beam max tokens")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Checkpoint every N samples")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in output directory")

    # Noise injection
    parser.add_argument('--noise-level', type=float, default=0.0,
                        help="Noise level for embedding perturbation (default: 0.0)")
    parser.add_argument('--noise-levels', type=float, nargs='+', default=None,
                        help="Sweep multiple noise levels (e.g., --noise-levels 0.001 0.01 0.1)")

    return parser.parse_args()


def main():
    args = parse_args()

    from run_metadata import track_experiment
    track_experiment(getattr(args, "output", None), args)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Encoder: {args.encoder_model}")
    print(f"Generator: {HARDCODED_GENERATOR} (hardcoded)")
    print(f"Use query prompt: {args.use_query_prompt}")
    print(f"Trust remote code: {args.trust_remote_code}")
    noise_levels = args.noise_levels if args.noise_levels else [args.noise_level]
    print(f"Noise levels: {noise_levels}")

    # Parse --kv-layers: '6-20', '8,10,12,14', 'middle', or None (all)
    kv_layers = None
    if args.kv_layers:
        if args.kv_layers == 'middle':
            # Auto-detect num_layers from the model config
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(HARDCODED_GENERATOR)
            n = cfg.n_layer if hasattr(cfg, 'n_layer') else cfg.num_hidden_layers
            kv_layers = list(range(n // 4, 3 * n // 4))  # middle 50%
            print(f"KV layers: 'middle' → {kv_layers} ({len(kv_layers)}/{n})")
        elif '-' in args.kv_layers and ',' not in args.kv_layers:
            lo, hi = args.kv_layers.split('-')
            kv_layers = list(range(int(lo), int(hi) + 1))
            print(f"KV layers: range {lo}-{hi} → {len(kv_layers)} layers")
        else:
            kv_layers = [int(x) for x in args.kv_layers.split(',')]
            print(f"KV layers: {kv_layers} ({len(kv_layers)} layers)")

    # Resolve output target. If --output ends in .csv, treat it as an explicit
    # output FILE (write the final result exactly there). Otherwise treat it as
    # a directory and use the dataset-derived auto-naming inside it. This avoids
    # the old footgun where `--output .../foo.csv` mkdir'd `foo.csv/` as a
    # directory and wrote a timestamped file inside (or errored if foo.csv
    # already existed as a file).
    _out_arg = Path(args.output)
    if _out_arg.suffix.lower() == ".csv":
        output_dir = _out_arg.parent
        output_file_override = _out_arg
    else:
        output_dir = _out_arg
        output_file_override = None
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    csv_path = get_dataset_path(args.dataset)
    print(f"Loading texts from: {csv_path}")
    texts = load_texts_from_csv(csv_path, text_column=args.text_column)
    print(f"Loaded {len(texts)} texts")

    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    # Initialize encoder
    if args.encoder_model == 'azure-openai':
        from azure_encoder import AzureOpenAIEncoder
        encoder = AzureOpenAIEncoder(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            noise_level=args.noise_level,
        )
        encoder_short_name = 'text-embedding-3-small'
    else:
        encoder = NoisySentenceTransformerWrapper(
            batch_size=512,
            model_name=args.encoder_model,
            trust_remote_code=args.trust_remote_code,
            noise_level=args.noise_level,
        )
        encoder_short_name = get_encoder_short_name(args.encoder_model)
    generator_name = HARDCODED_GENERATOR.split("/")[-1]

    # Initialize token boost
    retriever = None
    vocab_embeddings = None
    _lm_tokenizer = AutoTokenizer.from_pretrained(HARDCODED_GENERATOR)
    if args.boost_strength > 0:
        if "vocab" in args.boost_attributes:
            # Direct vocab boost: encode LLM vocab tokens through sentence encoder
            use_mask = args.vocab_boost_mask
            print(f"\n[Pre-computing vocab embeddings for token boost (mask={use_mask})]")
            MASK_PREFIX = "[MASK] [MASK] [MASK] [MASK] "
            _vocab_size = _lm_tokenizer.vocab_size
            allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\'-')
            vocab_texts = []
            vocab_valid = []  # tracks which indices are valid for boosting
            for i in range(_vocab_size):
                decoded = _lm_tokenizer.decode([i]).strip()
                is_valid = len(decoded) >= args.boost_min_token_length and all(c in allowed_chars for c in decoded)
                vocab_valid.append(is_valid)
                if use_mask:
                    vocab_texts.append(MASK_PREFIX + decoded if decoded else " ")
                else:
                    vocab_texts.append(decoded if decoded else " ")
            all_embs = []
            for vi in tqdm(range(0, len(vocab_texts), 512), desc="Encoding vocab"):
                batch = vocab_texts[vi:vi + 512]
                embs = encoder.get_logits(batch)
                if not isinstance(embs, torch.Tensor):
                    embs = torch.tensor(embs)
                all_embs.append(embs.cpu())
            vocab_embeddings = torch.cat(all_embs, dim=0)
            vocab_embeddings = F.normalize(vocab_embeddings, p=2, dim=-1)
            # Zero out invalid tokens so they never rank high
            for i, valid in enumerate(vocab_valid):
                if not valid:
                    vocab_embeddings[i] = 0.0
            print(f"  Vocab embeddings: {vocab_embeddings.shape} ({sum(vocab_valid)} valid tokens)")
            encoder.reset_token_count()
        else:
            # Knowledge bank retriever
            print("\n[Initializing attribute retriever for token boost]")
            from knowledge_bank.retriever import AttributeRetriever
            retriever = AttributeRetriever(args.encoder_model)
        encoder.reset_token_count()

    # =========================================================================
    # Sweep over noise levels
    # =========================================================================
    for noise_level in noise_levels:
        encoder.noise_level = noise_level
        noise_tag = f"_noise{noise_level}"

        if len(noise_levels) > 1:
            print(f"\n{'#'*80}")
            print(f"# Noise Level: {noise_level}")
            print(f"{'#'*80}")

        # =====================================================================
        # Handle resume logic
        # =====================================================================
        reconstructions = []
        start_idx = args.skip_samples

        if args.resume:
            print("\n[Checking for existing checkpoints]")
            checkpoint_file, latest_num = _find_latest_checkpoint(
                output_dir, dataset_name, generator_name, encoder_short_name, noise_level=noise_level
            )

            if checkpoint_file:
                reconstructions = _load_results(checkpoint_file, args.refine_top_n)
                if reconstructions:
                    last_sample_id = max(r[0] for r in reconstructions)
                    start_idx = last_sample_id + 1
                print(f"  Found checkpoint: {checkpoint_file}")
                print(f"  Loaded {len(reconstructions)} previous results")
                print(f"  Resuming from sample {start_idx}")
            else:
                print("  No checkpoint found, starting from scratch")
        elif args.skip_samples > 0:
            print(f"Skipping first {args.skip_samples} samples")

        # Prepare texts to process
        texts_to_process = texts[start_idx:]

        if args.max_samples:
            remaining = args.max_samples - len(reconstructions)
            if remaining <= 0:
                print(f"\n[Already processed {len(reconstructions)} samples, max_samples={args.max_samples}]")
                texts_to_process = []
            else:
                texts_to_process = texts_to_process[:remaining]

        print(f"\n[Running Inversion] noise_level={noise_level}")
        print(f"  Starting from sample {start_idx}, {len(texts_to_process)} samples to process")

        for i, text in enumerate(tqdm(texts_to_process, desc=f"Processing (noise={noise_level})")):
            actual_idx = start_idx + i

            print(f"\n{'='*80}")
            print(f"Sample {actual_idx}: {text[:100]}...")
            print(f"{'='*80}")

            # Reset token counter for this sample
            encoder.reset_token_count()

            if args.use_query_prompt:
                target_embedding = encoder.get_logits_noisy(text, prompt_name="query", sample_id=actual_idx)
            else:
                target_embedding = encoder.get_logits_noisy(text, sample_id=actual_idx)
            if target_embedding.dim() == 1:
                target_embedding = target_embedding.unsqueeze(0)
            if args.max_length_mode == "auto":
                _n_src = len(_lm_tokenizer.encode(text, add_special_tokens=False))
                target_seq_max_length = max(16, min(args.max_length_cap, round(_n_src * 1.15)))
            else:
                target_seq_max_length = args.max_sentence_length

            generator = Generator(
                seed=args.seed,
                generator_model=HARDCODED_GENERATOR,
                target_seq_max_length=target_seq_max_length,
                num_dummy_tokens=args.num_dummy_tokens,
                sentence_iterations=args.sentence_iterations,
                encoder_instance=encoder,
                use_query_prompt=args.use_query_prompt,
                learning_rate=args.learning_rate,
                sampling_top_k=args.sampling_top_k,
                cls_loss_scale=args.cls_loss_scale,
                fluency_loss_scale=args.fluency_loss_scale,
                loss_aggregation=args.loss_aggregation,
                similarity_temperature=args.similarity_temperature,
                entity_penalty=args.entity_penalty,
                short_token_penalty=args.short_token_penalty,
                min_token_length=args.min_token_length,
                num_loss_computations=args.num_loss_computations,
                kv_layers=kv_layers,
                boost_dedup=not args.no_boost_dedup,
            )

            # Build token boost
            sample_retrieved_json = ""
            if vocab_embeddings is not None:
                # Direct vocab boost mode
                t_emb = target_embedding.detach().cpu()
                if t_emb.dim() == 2:
                    t_emb = t_emb[0]
                t_emb = F.normalize(t_emb, p=2, dim=0)
                sims = torch.matmul(vocab_embeddings, t_emb)
                _, top_ids = sims.topk(min(args.num_boost_tokens, (sims > 0).sum().item()))
                boost = torch.zeros(generator.generator_model.config.vocab_size)
                for rank, tid in enumerate(top_ids.tolist()):
                    if tid < boost.size(0):
                        scale = 1.0 - 0.5 * (rank / max(1, args.num_boost_tokens - 1))
                        base = args.boost_strength * scale
                        decoded = _lm_tokenizer.decode([tid]).strip()
                        if decoded and decoded[0].isupper() and args.capitalize_boost != 1.0:
                            base *= args.capitalize_boost
                        boost[tid] = base
                generator.token_boost = boost
                # Debug: show boosted tokens with both boost score and underlying similarity
                top_ids_list = top_ids.tolist()
                num_boosted = (boost > 0).sum().item()
                top30 = [
                    (_lm_tokenizer.decode([tid]).strip(),
                     f"b={boost[tid].item():.2f}",
                     f"s={sims[tid].item():.3f}")
                    for tid in top_ids_list[:30]
                ]
                print(f"  [Vocab boost] {num_boosted} tokens. Top 30: {top30}")
            elif retriever is not None:
                # Knowledge bank retriever mode — also save retrieved values
                import json as _json
                retrieved_attrs = {}
                for attr in args.boost_attributes:
                    matches = retriever.retrieve(target_embedding, attr,
                                                 top_k=args.num_boost_tokens,
                                                 template=args.boost_template)
                    retrieved_attrs[attr] = [(v, round(s, 4)) for v, s in matches]
                sample_retrieved_json = _json.dumps(retrieved_attrs)

                generator.token_boost = retriever.get_boost_tensor(
                    target_embedding=target_embedding,
                    attributes=args.boost_attributes,
                    llm_tokenizer=_lm_tokenizer,
                    top_k=args.num_boost_tokens,
                    boost_strength=args.boost_strength,
                    vocab_size=generator.generator_model.config.vocab_size,
                    template=args.boost_template,
                )

            # Always populate knowledge bank JSON (even in vocab boost mode)
            if retriever is not None and not sample_retrieved_json:
                import json as _json
                retrieved_attrs = {}
                for attr in args.boost_attributes:
                    try:
                        matches = retriever.retrieve(target_embedding, attr,
                                                     top_k=5, template=args.boost_template)
                        retrieved_attrs[attr] = [(v, round(s, 4)) for v, s in matches]
                    except Exception:
                        pass
                if retrieved_attrs:
                    sample_retrieved_json = _json.dumps(retrieved_attrs)

            # Phase 1
            print("\n" + "=" * 60)
            print("PHASE 1: KV cache optimization")
            print("=" * 60)
            best_runs = generator.generate(target_embedding)

            # Get Phase 1 token count and reset for Phase 2
            kv_tokens = encoder.get_token_count()
            print(f"Phase 1 (KV) tokens: {kv_tokens}")
            encoder.reset_token_count()

            # Phase 2 (optional)
            refined_results = []
            beam_tokens = 0
            if args.enable_beam_search:
                print("\n" + "=" * 60)
                print("PHASE 2: Beam search refinement")
                print("=" * 60)

                for run_idx, run in enumerate(best_runs[: args.refine_top_n]):
                    if run["kv_state"] is None:
                        continue
                    prev_state = generator.load_kv_state(run["kv_state"])
                    refined_text, score = generator.refine_with_learned_context(
                        target_embedding=target_embedding,
                        beam_size=args.beam_size,
                        top_k=args.beam_top_k,
                        max_tokens=args.beam_max_tokens,
                    )
                    print(f"Refined {run_idx + 1}: {refined_text[:80]}... (score: {score:.4f})")
                    refined_results.append({"reconstruction": refined_text, "score": score})
                    generator.restore_kv_state(prev_state)

                # Get Phase 2 token count
                beam_tokens = encoder.get_token_count()
                print(f"Phase 2 (Beam) tokens: {beam_tokens}")
            else:
                print("\n[Phase 2 skipped - use --enable-beam-search to enable]")

            print(f"Total tokens: {kv_tokens + beam_tokens}")

            # Combine results
            phase1_texts = [run["reconstruction"] for run in best_runs[:10] if run["reconstruction"]]
            phase1_scores = [
                run["score"].item() if hasattr(run["score"], "item") else run["score"]
                for run in best_runs[:10]
                if run["reconstruction"]
            ]
            phase2_texts = [r["reconstruction"] for r in refined_results]
            phase2_scores = [r["score"] for r in refined_results]

            reconstructions.append((actual_idx, text, phase1_texts, phase1_scores, phase2_texts, phase2_scores, kv_tokens, beam_tokens, noise_level, sample_retrieved_json))

            # Checkpoint
            if len(reconstructions) % args.checkpoint_interval == 0:
                checkpoint_file = output_dir / f"{dataset_name}_{generator_name}_{encoder_short_name}{noise_tag}_checkpoint_{len(reconstructions)}.csv"
                _save_results(checkpoint_file, reconstructions, args.refine_top_n)
                print(f"Checkpoint saved: {checkpoint_file}")

        # Final save for this noise level. Use the explicit --output file when
        # one was given AND we're producing a single noise level (can't write
        # multiple noise levels to one filename); otherwise auto-name in the dir.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_file_override is not None and len(noise_levels) == 1:
            final_file = output_file_override
        else:
            final_file = output_dir / f"{dataset_name}_{generator_name}_{encoder_short_name}{noise_tag}_{timestamp}.csv"
        _save_results(final_file, reconstructions, args.refine_top_n)
        print(f"\nFinal results saved: {final_file}")

        ckpt_pattern = f"{dataset_name}_{generator_name}_{encoder_short_name}{noise_tag}_checkpoint_*.csv"
        ckpts = list(output_dir.glob(ckpt_pattern))
        for c in ckpts:
            try:
                c.unlink()
            except Exception as e:
                print(f"  [cleanup] could not remove {c.name}: {e}")
        if ckpts:
            print(f"[cleanup] removed {len(ckpts)} checkpoint file(s) matching {ckpt_pattern}")


def _save_results(filepath, reconstructions, refine_top_n):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        header = ["id", "source_text"]
        header += [f"c{j + 1}" for j in range(10)]
        header += [f"c{j + 1}_sim" for j in range(10)]
        header += [f"refined_{j + 1}" for j in range(refine_top_n)]
        header += [f"refined_{j + 1}_sim" for j in range(refine_top_n)]
        header += ["kv_tokens", "beam_tokens", "noise_level", "retrieved_attributes"]
        writer.writerow(header)

        for rec in reconstructions:
            row_id, text, p1_texts, p1_scores, p2_texts, p2_scores, kv_tokens, beam_tokens, nl = rec[:9]
            retrieved_json = rec[9] if len(rec) > 9 else ""
            row = [row_id, text]
            row += (p1_texts + [""] * 10)[:10]
            row += (p1_scores + [0.0] * 10)[:10]
            row += (p2_texts + [""] * refine_top_n)[:refine_top_n]
            row += (p2_scores + [0.0] * refine_top_n)[:refine_top_n]
            row += [kv_tokens, beam_tokens, nl, retrieved_json]
            writer.writerow(row)


def _load_results(filepath, refine_top_n):
    """Load results from a CSV checkpoint file."""
    reconstructions = []
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = int(row['id'])
            text = row['source_text']
            p1_texts = [row.get(f'c{j+1}', '') for j in range(10)]
            p1_scores = [float(row.get(f'c{j+1}_sim', 0.0) or 0.0) for j in range(10)]
            p2_texts = [row.get(f'refined_{j+1}', '') for j in range(refine_top_n)]
            p2_scores = [float(row.get(f'refined_{j+1}_sim', 0.0) or 0.0) for j in range(refine_top_n)]
            kv_tokens = int(row.get('kv_tokens', 0) or 0)
            beam_tokens = int(row.get('beam_tokens', 0) or 0)
            nl = float(row.get('noise_level', 0.0) or 0.0)
            retrieved_json = row.get('retrieved_attributes', '')
            reconstructions.append((row_id, text, p1_texts, p1_scores, p2_texts, p2_scores, kv_tokens, beam_tokens, nl, retrieved_json))
    return reconstructions


def _find_latest_checkpoint(output_dir, dataset_name, generator_name, encoder_short_name, noise_level=0.0):
    """Find the latest checkpoint file matching the current configuration."""
    import re

    noise_tag = f"_noise{noise_level}"
    pattern = f"{dataset_name}_{generator_name}_{encoder_short_name}{noise_tag}_checkpoint_*.csv"
    checkpoint_files = list(output_dir.glob(pattern))

    if not checkpoint_files:
        return None, -1

    checkpoint_info = []
    for f in checkpoint_files:
        match = re.search(r'_checkpoint_(\d+)\.csv$', str(f))
        if match:
            checkpoint_info.append((int(match.group(1)), f))

    if not checkpoint_info:
        return None, -1

    checkpoint_info.sort(key=lambda x: x[0], reverse=True)
    latest_num, latest_file = checkpoint_info[0]

    return latest_file, latest_num


if __name__ == "__main__":
    main()
