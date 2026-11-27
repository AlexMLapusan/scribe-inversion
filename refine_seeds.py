#!/usr/bin/env python3
"""
Seed Refinement via LLM-guided Beam Search

This script takes seed texts from a CSV and refines them using the Stage 2
paraphrase-based refinement from ZSInvert. It prompts an LLM with
"write a sentence similar to: <seed>" and uses beam search with cosine
similarity scoring to find better reconstructions.

Usage:
    python refine_seeds.py --input data.csv --column seed_text --encoder gte --llm Qwen/Qwen2.5-0.5B-Instruct

Required packages:
    pip install torch transformers sentence-transformers pandas tqdm
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# ENCODER MAPPING
# =============================================================================
ENCODERS = {
    'gte': 'thenlper/gte-base',
    'gte-qwen': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    'gtr': 'sentence-transformers/gtr-t5-base',
    'contriever': 'facebook/contriever',
    'azure-openai': 'text-embedding-3-small',
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class Candidate:
    """Represents one beam-search candidate."""
    token_ids: List[int] = field(default_factory=list)
    score: float = 0.0
    seq_str: str = ""
    cos_sim: float = 0.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
NOISE_BASE_SEED = 1_234_567


def _deterministic_noise(dim, sample_id, device, dtype):
    """Reproducible Gaussian noise vector of length `dim` for one sample.

    Seeded by NOISE_BASE_SEED + sample_id and drawn on CPU so the identical
    vector is reproduced in every pipeline phase (KV/HC/refine) regardless of
    which GPU runs it. Must match the implementation in invert_gpt2.py and
    hill_climb.py so all phases attack the SAME fixed noisy target.
    """
    if sample_id is None:
        return torch.randn(dim, device=device, dtype=dtype)
    g = torch.Generator()
    g.manual_seed(NOISE_BASE_SEED + int(sample_id))
    return torch.randn(dim, generator=g).to(device=device, dtype=dtype)


def count_tokens(encoder, texts):
    """Count tokens for a list of texts using the encoder's tokenizer."""
    if texts is None or (isinstance(texts, float) and pd.isna(texts)):
        return 0
    if isinstance(texts, str):
        texts = [texts]
    total = 0
    tokenizer = encoder.tokenizer
    for text in texts:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            continue
        tokens = tokenizer.encode(str(text), add_special_tokens=True)
        total += len(tokens)
    return total


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 50, top_p: float = 1.0) -> torch.Tensor:
    """Filter tokens by top-k and top-p constraints."""
    probs = torch.softmax(logits, dim=-1)

    # Top-K filtering
    topk_probs, topk_indices = torch.topk(probs, min(top_k, probs.size(-1)))

    # Top-P (nucleus) filtering
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p_cutoff = torch.sum(cumulative_probs <= top_p).item()
    if top_p_cutoff < 1:
        top_p_cutoff = 1
    top_p_indices = sorted_indices[:top_p_cutoff]

    # Intersection of top-k and top-p
    top_k_set = set(topk_indices.tolist())
    top_p_set = set(top_p_indices.tolist())
    intersection_set = top_k_set.intersection(top_p_set)

    if len(intersection_set) == 0:
        return topk_indices

    intersection_indices = torch.tensor(list(intersection_set), dtype=torch.long)
    intersection_probs = probs[intersection_indices]
    sorted_intersection_probs, sorted_order = torch.sort(intersection_probs, descending=True)
    intersection_indices = intersection_indices[sorted_order.cpu()]

    return intersection_indices


# =============================================================================
# COSINE SIMILARITY SCORER
# =============================================================================
class CosineSimilarityScorer:
    """Scores candidates by cosine similarity with target embedding."""

    def __init__(self, encoder: SentenceTransformer, target_embedding: torch.Tensor):
        self.encoder = encoder
        self.target_embedding = target_embedding  # Shape: [1, hidden_dim]
        self.total_tokens_embedded = 0

    def reset_token_count(self):
        """Reset the token counter."""
        self.total_tokens_embedded = 0

    def get_token_count(self):
        """Get the current token count."""
        return self.total_tokens_embedded

    def score_candidates(self, candidates: List[Candidate]) -> List[float]:
        """Score all candidates by cosine similarity with target embedding."""
        if len(candidates) == 0:
            return []

        texts = [c.seq_str for c in candidates]

        # Count tokens before encoding
        self.total_tokens_embedded += count_tokens(self.encoder, texts)

        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        cos_sims = torch.matmul(embeddings, self.target_embedding.t()).squeeze(-1)
        cos_sims = cos_sims.cpu().tolist()

        for i, c in enumerate(candidates):
            c.cos_sim = cos_sims[i]
            c.score = cos_sims[i]

        return cos_sims


# =============================================================================
# LLM WRAPPER
# =============================================================================
class LLMWrapper:
    """Wrapper for LLM that generates next-token proposals."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda:0",
        system_message: str = None,
        few_shot_examples: List[Tuple[str, str]] = None
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self._setup_chat_format(system_message, few_shot_examples)

    def _setup_chat_format(self, system_message: str = None,
                           few_shot_examples: List[Tuple[str, str]] = None):
        """Setup chat format prefix and suffix tokens.

        Args:
            system_message: Optional system prompt.
            few_shot_examples: Optional list of (user, assistant) example pairs.
        """
        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        if few_shot_examples:
            for user_msg, assistant_msg in few_shot_examples:
                messages.append({'role': 'user', 'content': user_msg})
                messages.append({'role': 'assistant', 'content': assistant_msg})
        messages.append({'role': 'user', 'content': ''})

        try:
            full_template = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True
            )
            user_only = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True
            )
            self.chat_prefix = user_only[:-1] if user_only else []
            self.chat_suffix = full_template[len(user_only)-1:] if len(full_template) > len(user_only) else []
        except Exception:
            self.chat_prefix = []
            self.chat_suffix = []

    def prepare_input(self, prompt_tokens: List[int], generated_tokens: List[int]) -> List[int]:
        """Prepare full input for the LLM."""
        return self.chat_prefix + prompt_tokens + self.chat_suffix + generated_tokens

    def get_next_token_candidates(
        self,
        prompt_tokens: List[int],
        batch_tokens: List[List[int]],
        top_k: int = 30,
        top_p: float = 1.0,
        token_boost: torch.Tensor = None,
    ) -> List[List[Tuple[int, float]]]:
        """Get next token candidates for a batch of beams."""
        full_inputs = [
            self.prepare_input(prompt_tokens, tokens)
            for tokens in batch_tokens
        ]

        max_len = max(len(inp) for inp in full_inputs)
        padded_inputs = [
            inp + [self.tokenizer.pad_token_id or 0] * (max_len - len(inp))
            for inp in full_inputs
        ]

        input_ids = torch.tensor(padded_inputs, device=self.device)
        attention_mask = torch.tensor([
            [1] * len(full_inputs[i]) + [0] * (max_len - len(full_inputs[i]))
            for i in range(len(full_inputs))
        ], device=self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        logits_list = []
        for i, inp in enumerate(full_inputs):
            last_pos = len(inp) - 1
            logits_list.append(outputs.logits[i, last_pos, :])

        # Mask special tokens
        special_tokens_to_mask = []
        for special in ['<|im_end|>', '<|end_header_id|>', '<|start_header_id|>',
                       '<|eot_id|>', '<|eom_id|>', '<|python_tag|>']:
            try:
                tokens = self.tokenizer.encode(special, add_special_tokens=False)
                if len(tokens) == 1:
                    special_tokens_to_mask.append(tokens[0])
            except:
                pass

        if self.tokenizer.eos_token_id is not None:
            special_tokens_to_mask.append(self.tokenizer.eos_token_id)

        results = []
        for logits in logits_list:
            for token_id in special_tokens_to_mask:
                logits[token_id] = -1e10

            if token_boost is not None:
                logits = logits + token_boost

            filtered_indices = top_k_top_p_filtering(logits, top_k, top_p)
            log_probs = F.log_softmax(logits, dim=-1)
            filtered_log_probs = log_probs[filtered_indices]

            sorted_vals, sorted_idx = torch.sort(filtered_log_probs, descending=True)
            sorted_indices = filtered_indices[sorted_idx.cpu()]

            results.append(list(zip(
                sorted_indices.tolist(),
                sorted_vals.tolist()
            )))

        return results


# =============================================================================
# BEAM SEARCH
# =============================================================================
class BeamSearch:
    """Beam search with cosine similarity scoring."""

    def __init__(
        self,
        llm_wrapper: LLMWrapper,
        scorer: CosineSimilarityScorer,
        beam_width: int = 30,
        max_steps: int = 32,
        top_k: int = 30,
        top_p: float = 1.0,
        token_boost: torch.Tensor = None,
    ):
        self.llm = llm_wrapper
        self.scorer = scorer
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.top_k = top_k
        self.top_p = top_p
        self.tokenizer = llm_wrapper.tokenizer
        self.token_boost = token_boost

    def search(self, prompt_tokens: List[int], verbose: bool = False) -> Candidate:
        """Run beam search to find the best sequence."""
        candidates = [Candidate(token_ids=[], seq_str="", score=0.0)]
        best_full_candidate = None

        iterator = tqdm(range(self.max_steps), desc="Beam Search", leave=False) if verbose else range(self.max_steps)

        for step in iterator:
            all_candidates: List[Candidate] = []
            batch_tokens = [cand.token_ids for cand in candidates]

            batch_proposals = self.llm.get_next_token_candidates(
                prompt_tokens,
                batch_tokens,
                top_k=self.top_k,
                top_p=self.top_p,
                token_boost=self.token_boost,
            )

            for i, cand in enumerate(candidates):
                for token_id, log_prob in batch_proposals[i]:
                    new_token_ids = cand.token_ids + [token_id]
                    new_seq_str = self.tokenizer.decode(new_token_ids)

                    new_candidate = Candidate(
                        token_ids=new_token_ids,
                        seq_str=new_seq_str,
                        score=0.0
                    )
                    all_candidates.append(new_candidate)

            self.scorer.score_candidates(all_candidates)
            all_candidates.sort(key=lambda c: c.score, reverse=True)

            for cand in all_candidates:
                if cand.seq_str and cand.seq_str.strip():
                    last_char = cand.seq_str.strip()[-1]
                    if last_char in '.?!\n':
                        if best_full_candidate is None or cand.score > best_full_candidate.score:
                            best_full_candidate = cand

            candidates = all_candidates[:self.beam_width]

        if best_full_candidate is not None:
            return best_full_candidate
        return candidates[0]


# =============================================================================
# SEED REFINER
# =============================================================================
REFINE_SYSTEM_PROMPT = (
    "You reconstruct original sentences from noisy copies. "
    "Output only the reconstructed sentence. "
    "No quotes, no labels, no explanation."
)

# One-shot example matching real embedding-inversion output difficulty:
# fragmented names, mangled medical terms, random insertions, no structure.
REFINE_FEW_SHOT = [(
    "Noisy copies of a sentence:\n"
    "1. Chen Robert cardiac insulin diabetes MI heart attack drug Hep patient ICU drip Cheng Rob acute myocard died hospital infarct\n"
    "2. Rob Chen diabetic insulin MI patient ICUs cardiac heparinstol infarction acute heart CH diabetes Rob Chenn myocardiac arrested\n"
    "3. Chen Robt acute MI diabetic myocardiac infraction admitted Chen cardiac ICU hepardin insulin drip type diabetes heart Chen hospital\n"
    "Reconstructed sentence:",
    "Robert Chen, 62, admitted with acute myocardial infarction and type 2 diabetes, started on heparin drip and insulin protocol in the cardiac ICU."
)]


class SeedRefiner:
    """
    Refines seed texts using LLM-guided beam search.

    Uses multi-candidate or single-seed prompts to guide
    generation toward high cosine similarity with a target embedding.
    """

    def __init__(
        self,
        encoder: SentenceTransformer,
        llm_wrapper: LLMWrapper,
        beam_width: int = 30,
        max_steps: int = 32,
        top_k: int = 30,
        top_p: float = 1.0,
        n_iter: int = 1,
        token_boost: torch.Tensor = None,
    ):
        self.encoder = encoder
        self.llm_wrapper = llm_wrapper
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.top_k = top_k
        self.top_p = top_p
        self.n_iter = n_iter
        self.tokenizer = llm_wrapper.tokenizer
        self.token_boost = token_boost

    @staticmethod
    def _build_multi_candidate_prompt(seed_texts: List[str]) -> str:
        """Build a prompt from multiple noisy candidate reconstructions.

        Uses a concise format optimized for small LLMs (Qwen2.5-0.5B+):
        numbered candidates followed by a clear generation trigger.
        No quotes around candidates to avoid them leaking into output.
        """
        lines = ["Noisy copies of a sentence:"]
        for i, text in enumerate(seed_texts, 1):
            lines.append(f"{i}. {text.strip()}")
        lines.append("Reconstructed sentence:")
        return "\n".join(lines)

    @staticmethod
    def _clean_output(text: str) -> str:
        """Strip quotes and artifacts from beam search output."""
        text = text.strip()
        # Strip wrapping quotes
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1].strip()
        # Strip leading quote if present
        if text.startswith('"') or text.startswith("'"):
            text = text[1:].strip()
        return text

    def refine(
        self,
        seed_text: str,
        target_embedding: torch.Tensor,
        verbose: bool = False,
        extra_seeds: List[str] = None
    ) -> Tuple[str, float, int]:
        """
        Refine a seed text to better match the target embedding.

        Args:
            seed_text: Primary seed text to refine
            target_embedding: Target embedding to match (normalized, shape [hidden_dim])
            verbose: Whether to print progress
            extra_seeds: Additional candidate texts for multi-candidate prompting.
                         Used in iteration 1 to give the LLM richer context;
                         subsequent iterations use the single refined output.

        Returns:
            Tuple of (refined_text, cosine_similarity, total_tokens)
        """
        # Ensure target embedding is 2D
        if target_embedding.dim() == 1:
            target_embedding = target_embedding.unsqueeze(0)

        scorer = CosineSimilarityScorer(self.encoder, target_embedding)

        x_current = seed_text
        best_result_text = seed_text
        best_result_score = -1.0

        for iteration in range(self.n_iter):
            if verbose:
                print(f"  Refinement iteration {iteration + 1}/{self.n_iter}")

            # First iteration with multiple candidates: use multi-candidate prompt
            # Subsequent iterations: single-seed prompt with latest refined output
            if iteration == 0 and extra_seeds:
                all_seeds = [seed_text] + [s for s in extra_seeds if s and str(s).strip()]
                refine_prompt = self._build_multi_candidate_prompt(all_seeds)
            else:
                refine_prompt = f"write a sentence similar to: {x_current}"

            if verbose:
                print(f"    Prompt: {refine_prompt[:120]}...")

            refine_prompt_tokens = self.tokenizer.encode(refine_prompt, add_special_tokens=False)

            beam_search = BeamSearch(
                llm_wrapper=self.llm_wrapper,
                scorer=scorer,
                beam_width=self.beam_width,
                max_steps=self.max_steps,
                top_k=self.top_k,
                top_p=self.top_p,
                token_boost=self.token_boost,
            )

            x_refined = beam_search.search(refine_prompt_tokens, verbose=verbose)
            x_current = self._clean_output(x_refined.seq_str)

            if x_refined.cos_sim > best_result_score:
                best_result_text = x_current
                best_result_score = x_refined.cos_sim

            if verbose:
                print(f"    Result: '{x_current[:80]}...' (cos_sim={x_refined.cos_sim:.4f})")

        total_tokens = scorer.get_token_count()
        return best_result_text, best_result_score, total_tokens


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Refine seed texts via LLM-guided beam search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python refine_seeds.py --input seeds.csv --column inversion --target-column text --encoder gte
    python refine_seeds.py --input data.csv --column seed --target-column original --encoder gte-qwen --beam-width 50
    python refine_seeds.py --input results.csv --seed-columns c1 c2 c3 c4 c5 --target-column source_text --encoder azure-openai
    python refine_seeds.py --input results.csv --column hypothesis --target-column reference --n-iter 3
        """
    )

    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help="Input CSV file path")
    parser.add_argument('--column', type=str, default=None,
                        help="Single column containing seed texts (use --seed-columns for multi-candidate)")
    parser.add_argument('--seed-columns', type=str, nargs='+', default=None,
                        help="Multiple columns for multi-candidate refinement (e.g., --seed-columns c1 c2 c3 c4 c5). "
                             "First column is the primary seed; the rest provide additional context.")
    parser.add_argument('--target-column', type=str, default=None,
                        help="Column name containing target texts for embedding (if not provided, uses primary seed column as target)")
    parser.add_argument('--output', type=str, default=None,
                        help="Output CSV file path (default: overwrites input with new columns)")
    parser.add_argument('--resume', action='store_true',
                        help="If output CSV exists, load it and skip rows where the "
                             "refined column is already populated.")
    parser.add_argument('--refined-column', type=str, default='refined_text',
                        help="Name for the output refined text column (default: refined_text)")
    parser.add_argument('--score-column', type=str, default='refined_cos_sim',
                        help="Name for the output similarity score column (default: refined_cos_sim)")

    # Model selection
    parser.add_argument('--encoder', type=str, default='gte',
                        help="Encoder model: gte, gte-qwen, gtr, contriever, azure-openai, or full HuggingFace path (default: gte)")
    parser.add_argument('--llm', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help="LLM for token generation (default: Qwen/Qwen2.5-0.5B-Instruct)")

    # Algorithm parameters
    parser.add_argument('--beam-width', type=int, default=30,
                        help="Beam width (default: 30)")
    parser.add_argument('--top-k', type=int, default=30,
                        help="Top-k tokens per step (default: 30)")
    parser.add_argument('--top-p', type=float, default=1.0,
                        help="Top-p (nucleus) sampling threshold (default: 1.0)")
    parser.add_argument('--max-steps', type=int, default=32,
                        help="Max sequence length (default: 32)")
    parser.add_argument('--max-length-mode', type=str, default='fixed', choices=['fixed', 'auto'],
                        help="'fixed' uses --max-steps for every sample; 'auto' sets a per-sample length "
                             "from the target text's generator-token count (round(1.15x), clamped to [16, cap])")
    parser.add_argument('--max-length-cap', type=int, default=96,
                        help="Upper bound on the per-sample reconstruction length in 'auto' mode")
    parser.add_argument('--n-iter', type=int, default=1,
                        help="Number of refinement iterations per seed (default: 1)")

    # Token boost (vocab similarity, knowledge bank, or both)
    parser.add_argument('--boost-strength', type=float, default=0,
                        help="Logit boost strength (0=disable)")
    parser.add_argument('--boost-mode', type=str, default='bank',
                        choices=['vocab', 'bank', 'both'],
                        help="'vocab'=top-K vocab tokens by similarity, "
                             "'bank'=knowledge bank attributes, "
                             "'both'=combine vocab and bank via max")
    parser.add_argument('--boost-attributes', nargs='+', default=['first_names', 'last_names', 'diagnoses'],
                        help="Knowledge bank attributes to boost (used in bank/both mode)")
    parser.add_argument('--boost-template', type=str, default=None,
                        help="Template for boost: None=per-attribute, 'mask', 'bare'")
    parser.add_argument('--num-boost-tokens', type=int, default=200,
                        help="Top-K vocab tokens (vocab mode) or matches per attribute (bank mode)")
    parser.add_argument('--boost-min-token-length', type=int, default=2,
                        help="Min decoded token length for vocab boost (excludes 'he','on','it' etc)")
    parser.add_argument('--capitalize-boost', type=float, default=1.0,
                        help="Extra multiplier for capitalized tokens in vocab mode (1.0=disable)")
    parser.add_argument('--print-boost-top-k', type=int, default=15,
                        help="How many top boosted tokens to print per sample (0=disable)")

    # Processing options
    parser.add_argument('--max-samples', type=int, default=None,
                        help="Max number of samples to process")
    parser.add_argument('--skip-samples', type=int, default=0,
                        help="Skip first N samples")
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help="Save checkpoint every N samples (default: 10)")

    # Hardware
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="Device for computation (default: cuda:0)")

    parser.add_argument('--noise-level', type=float, default=0.0,
                        help="Gaussian sigma added to the target embedding (defense "
                             "simulation, matches the ZSInvert noise threat model).")

    # Misc
    parser.add_argument('--verbose', action='store_true',
                        help="Print detailed progress")

    args = parser.parse_args()

    from run_metadata import track_experiment
    track_experiment(args.output, args)

    # =========================================================================
    # Validate column args
    # =========================================================================
    if not args.column and not args.seed_columns:
        parser.error("Either --column or --seed-columns is required")
    if args.column and args.seed_columns:
        parser.error("Use --column or --seed-columns, not both")

    # Determine primary column and extra seed columns
    if args.seed_columns:
        primary_column = args.seed_columns[0]
        extra_columns = args.seed_columns[1:]
    else:
        primary_column = args.column
        extra_columns = []

    # =========================================================================
    # Load CSV (resume-aware: prefer existing output if --resume)
    # =========================================================================
    resume_source = None
    if args.resume and args.output and os.path.exists(args.output):
        resume_source = args.output
    if resume_source:
        print(f"[Resume] Loading existing output: {resume_source}")
        df = pd.read_csv(resume_source)
    else:
        print(f"Loading CSV: {args.input}")
        df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")
    if resume_source and args.refined_column in df.columns:
        n_done = int(df[args.refined_column].notna().sum())
        print(f"  {n_done} rows already refined (will be skipped)")

    if primary_column not in df.columns:
        raise ValueError(f"Column '{primary_column}' not found. Available: {list(df.columns)}")
    for col in extra_columns:
        if col not in df.columns:
            raise ValueError(f"Seed column '{col}' not found. Available: {list(df.columns)}")

    target_column = args.target_column if args.target_column else primary_column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")

    # Apply skip and max samples
    start_idx = args.skip_samples
    end_idx = len(df) if args.max_samples is None else min(start_idx + args.max_samples, len(df))

    print(f"Processing rows {start_idx} to {end_idx - 1}")

    # =========================================================================
    # Load models
    # =========================================================================
    print("\n[Loading Encoder]")
    if args.encoder in ENCODERS:
        encoder_name = ENCODERS[args.encoder]
    else:
        encoder_name = args.encoder  # Assume it's a full path

    if args.encoder == 'azure-openai':
        from azure_encoder import AzureOpenAIEncoder
        encoder = AzureOpenAIEncoder(
            device=args.device,
        )
    else:
        trust_remote_code = 'qwen' in encoder_name.lower()
        encoder = SentenceTransformer(encoder_name, device=args.device, trust_remote_code=trust_remote_code)
    print(f"  Loaded: {encoder_name}")

    print("\n[Loading LLM]")
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm,
        torch_dtype=torch.bfloat16
    ).eval()
    print(f"  Loaded: {args.llm}")

    # Only use system prompt / few-shot for multi-candidate mode
    if extra_columns:
        llm_wrapper = LLMWrapper(
            model, tokenizer, device=args.device,
            system_message=REFINE_SYSTEM_PROMPT,
            few_shot_examples=REFINE_FEW_SHOT
        )
    else:
        llm_wrapper = LLMWrapper(model, tokenizer, device=args.device)

    # =========================================================================
    # Create refiner
    # =========================================================================
    refiner = SeedRefiner(
        encoder=encoder,
        llm_wrapper=llm_wrapper,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        top_k=args.top_k,
        top_p=args.top_p,
        n_iter=args.n_iter
    )

    # =========================================================================
    # Print configuration
    # =========================================================================
    print("\n" + "=" * 80)
    print("Seed Refinement Configuration")
    print("=" * 80)
    print(f"Encoder: {encoder_name}")
    print(f"LLM: {args.llm}")
    print(f"Parameters: beam_width={args.beam_width}, top_k={args.top_k}, "
          f"top_p={args.top_p}, max_steps={args.max_steps}, n_iter={args.n_iter}")
    print(f"Device: {args.device}")
    if extra_columns:
        print(f"Seed columns: {primary_column} + {extra_columns} (multi-candidate)")
    else:
        print(f"Input column: {primary_column}")
    print(f"Target column: {target_column}")
    print(f"Output columns: {args.refined_column}, {args.score_column}")
    print("=" * 80)

    # =========================================================================
    # Pre-compute target embeddings
    # =========================================================================
    print("\n[Pre-computing target embeddings]")
    target_texts = df[target_column].iloc[start_idx:end_idx].tolist()

    # Count tokens for each target text (will be added to per-sample counts)
    target_token_counts = [count_tokens(encoder, text) for text in target_texts]

    target_embeddings = encoder.encode(
        target_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    if args.noise_level > 0:
        target_ids = (df['id'].iloc[start_idx:end_idx].tolist()
                      if 'id' in df.columns else list(range(start_idx, end_idx)))
        for j, sid in enumerate(target_ids):
            noise = _deterministic_noise(
                target_embeddings[j].shape[-1], sid,
                target_embeddings.device, target_embeddings.dtype)
            target_embeddings[j] = target_embeddings[j] + noise * args.noise_level
        target_embeddings = F.normalize(target_embeddings, p=2, dim=-1)
        print(f"  Applied deterministic per-sample Gaussian noise (sigma={args.noise_level}) + re-normalized target embeddings")
    print(f"  Computed {len(target_embeddings)} embeddings")

    # =========================================================================
    # Initialize output columns if they don't exist
    # =========================================================================
    if args.refined_column not in df.columns:
        df[args.refined_column] = None
    if args.score_column not in df.columns:
        df[args.score_column] = None
    if 'refined_tokens' not in df.columns:
        df['refined_tokens'] = None

    # =========================================================================
    # Run refinement
    # =========================================================================
    print("\n[Running Refinement]")

    output_path = args.output if args.output else args.input
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # =========================================================================
    # Initialize boost: knowledge bank retriever and/or vocab embeddings
    # =========================================================================
    retriever = None
    vocab_embeddings = None
    vocab_decoded = None  # parallel list of decoded token strings (for printing)

    if args.boost_strength > 0 and args.boost_mode in ('bank', 'both'):
        print("\n[Initializing attribute retriever for token boost]")
        from knowledge_bank.retriever import AttributeRetriever
        encoder_full_name = ENCODERS.get(args.encoder, args.encoder)
        retriever = AttributeRetriever(encoder_full_name)

    if args.boost_strength > 0 and args.boost_mode in ('vocab', 'both'):
        print(f"\n[Pre-computing vocab embeddings for vocab boost (min_len={args.boost_min_token_length})]")
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\'-')
        llm_vocab_size = tokenizer.vocab_size
        mask_prefix = "[MASK] [MASK] [MASK] [MASK] "
        vocab_texts_for_enc = []
        vocab_decoded = []
        vocab_valid = []
        for i in range(llm_vocab_size):
            decoded = tokenizer.decode([i]).strip()
            vocab_decoded.append(decoded)
            is_valid = (
                len(decoded) >= args.boost_min_token_length
                and all(c in allowed_chars for c in decoded)
            )
            vocab_valid.append(is_valid)
            vocab_texts_for_enc.append(mask_prefix + decoded if decoded else " ")

        all_embs = []
        for vi in tqdm(range(0, len(vocab_texts_for_enc), 512), desc="Encoding vocab"):
            batch = vocab_texts_for_enc[vi:vi + 512]
            embs = encoder.encode(batch, convert_to_tensor=True,
                                  normalize_embeddings=True, show_progress_bar=False)
            all_embs.append(embs.cpu())
        vocab_embeddings = torch.cat(all_embs, dim=0)
        # Zero out invalid tokens so they never rank high in similarity search
        for i, valid in enumerate(vocab_valid):
            if not valid:
                vocab_embeddings[i] = 0.0
        print(f"  Vocab embeddings: {vocab_embeddings.shape} ({sum(vocab_valid)} valid tokens)")

    llm_vocab_size = model.config.vocab_size

    for i, idx in enumerate(tqdm(range(start_idx, end_idx), desc="Refining")):
        if args.resume and args.refined_column in df.columns and pd.notna(df.at[idx, args.refined_column]):
            continue
        seed_text = df.at[idx, primary_column]
        target_embedding = target_embeddings[i]

        if pd.isna(seed_text) or not str(seed_text).strip():
            print(f"  Skipping row {idx}: empty seed")
            continue

        seed_text = str(seed_text)

        # ================================================================
        # Build per-sample token boost (vocab / bank / both)
        # ================================================================
        sample_boost = None
        sims_vocab = None  # cached for printing later
        if args.boost_strength > 0:
            # Vocab boost
            if vocab_embeddings is not None:
                t_emb = target_embedding.detach().cpu()
                if t_emb.dim() == 2:
                    t_emb = t_emb[0]
                t_emb = F.normalize(t_emb, p=2, dim=0)
                sims_vocab = torch.matmul(vocab_embeddings, t_emb)
                k = min(args.num_boost_tokens, int((sims_vocab > 0).sum().item()))
                if k > 0:
                    _, top_ids = sims_vocab.topk(k)
                    sample_boost = torch.zeros(llm_vocab_size)
                    for rank, tid in enumerate(top_ids.tolist()):
                        if tid < llm_vocab_size:
                            scale = 1.0 - 0.5 * (rank / max(1, args.num_boost_tokens - 1))
                            base = args.boost_strength * scale
                            decoded = vocab_decoded[tid] if vocab_decoded else tokenizer.decode([tid]).strip()
                            if decoded and decoded[0].isupper() and args.capitalize_boost != 1.0:
                                base *= args.capitalize_boost
                            sample_boost[tid] = base

            # Knowledge bank boost (combined via max for 'both' mode)
            if retriever is not None:
                bank_boost = retriever.get_boost_tensor(
                    target_embedding=target_embedding,
                    attributes=args.boost_attributes,
                    llm_tokenizer=tokenizer,
                    top_k=args.num_boost_tokens,
                    boost_strength=args.boost_strength,
                    vocab_size=llm_vocab_size,
                    template=args.boost_template,
                )
                if sample_boost is None:
                    sample_boost = bank_boost.cpu()
                else:
                    sample_boost = torch.max(sample_boost, bank_boost.cpu())

            # Move boost to the LLM's device so it can be added to logits during beam search
            if sample_boost is not None:
                sample_boost = sample_boost.to(args.device)

        refiner.token_boost = sample_boost

        # Gather extra seed texts for multi-candidate prompting
        extra_seeds = []
        for col in extra_columns:
            val = df.at[idx, col]
            if not pd.isna(val) and str(val).strip():
                extra_seeds.append(str(val))

        # ================================================================
        # Tracking: compute seed similarity for delta reporting
        # ================================================================
        with torch.no_grad():
            seed_emb = encoder.encode(
                [seed_text], convert_to_tensor=True,
                normalize_embeddings=True, show_progress_bar=False
            )
            t_emb_2d = target_embedding.unsqueeze(0) if target_embedding.dim() == 1 else target_embedding
            seed_sim = torch.nn.functional.cosine_similarity(seed_emb, t_emb_2d, dim=1).item()

        print(f"\n[Sample {idx}]")
        print(f"  Seed:    {seed_text[:90]}")
        print(f"           sim={seed_sim:.4f}")
        if extra_seeds:
            print(f"  Extra seeds: {len(extra_seeds)} candidates")

        # Print top boosted tokens with similarity scores (sorted by boost desc)
        if sample_boost is not None and args.print_boost_top_k > 0:
            boosted_idx = (sample_boost > 0).nonzero(as_tuple=True)[0].tolist()
            ranked = sorted(boosted_idx, key=lambda t: -sample_boost[t].item())
            top_print = []
            for tid in ranked[:args.print_boost_top_k]:
                tok = vocab_decoded[tid] if vocab_decoded else tokenizer.decode([tid]).strip()
                b = sample_boost[tid].item()
                if sims_vocab is not None:
                    s = sims_vocab[tid].item()
                    top_print.append(f"({tok!r} b={b:.2f} s={s:.3f})")
                else:
                    top_print.append(f"({tok!r} b={b:.2f})")
            print(f"  Boost top {args.print_boost_top_k}/{len(boosted_idx)}: {' '.join(top_print)}")

        if args.max_length_mode == 'auto':
            _n_src = len(tokenizer.encode(str(target_texts[i]), add_special_tokens=False))
            refiner.max_steps = max(16, min(args.max_length_cap, round(_n_src * 1.15)))

        refined_text, cos_sim, refinement_tokens = refiner.refine(
            seed_text=seed_text,
            target_embedding=target_embedding,
            verbose=args.verbose,
            extra_seeds=extra_seeds if extra_seeds else None
        )

        refined_tokens = target_token_counts[i] + refinement_tokens

        df.at[idx, args.refined_column] = refined_text
        df.at[idx, args.score_column] = cos_sim
        df.at[idx, 'refined_tokens'] = refined_tokens

        delta = cos_sim - seed_sim
        delta_marker = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  Refined: {refined_text[:90]}")
        print(f"           sim={cos_sim:.4f}  {delta_marker} Δ={delta:+.4f}  refined_tokens={refined_tokens}")

        # Checkpoint
        if (i + 1) % args.checkpoint_interval == 0:
            df.to_csv(output_path, index=False)
            print(f"  Checkpoint saved ({i + 1} samples processed)")

    # =========================================================================
    # Final save
    # =========================================================================
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    processed_df = df.iloc[start_idx:end_idx]
    valid_scores = processed_df[args.score_column].dropna()

    if len(valid_scores) > 0:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total rows processed: {len(valid_scores)}")
        print(f"Average Cosine Similarity: {valid_scores.mean():.4f}")
        print(f"Min Cosine Similarity: {valid_scores.min():.4f}")
        print(f"Max Cosine Similarity: {valid_scores.max():.4f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
