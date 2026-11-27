#!/usr/bin/env python3
"""
Hill-climb post-processing for embedding inversion.

Takes a CSV with seed texts (from Phase 1 KV cache, Phase 2 refine_seeds, or any
source), and for each row runs black-box local token substitution to directly
maximize cosine similarity to the target embedding.

Key feature vs. the in-process version in invert_gpt2.py: excludes tokens whose
case/whitespace variants are already in the seed sentence, forcing the search
to propose genuinely new vocabulary rather than reallocating existing tokens.

Usage:
    # On Phase-1 c1 output
    python hill_climb.py --input outputs/ours/newer/mimic/contriever.csv \\
        --output outputs/ours/newer/mimic/contriever_hc.csv \\
        --column c1 --target-column source_text --encoder contriever

    # On Phase-2 refined output (the main comparison we want)
    python hill_climb.py --input outputs/ours/newer/refined/mimic/contriever.csv \\
        --output outputs/ours/newer/refined/mimic/contriever_hc.csv \\
        --column refined_text --target-column source_text --encoder contriever
"""

import argparse
import hashlib
import random
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


ENCODERS = {
    'gte': 'thenlper/gte-base',
    'gte-qwen': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    'gtr': 'sentence-transformers/gtr-t5-base',
    'contriever': 'facebook/contriever',
}


NOISE_BASE_SEED = 1_234_567


def _deterministic_noise(dim, sample_id, device, dtype):
    """Reproducible Gaussian noise vector of length `dim` for one sample.

    Seeded by NOISE_BASE_SEED + sample_id and drawn on CPU so the identical
    vector is reproduced in every pipeline phase (KV/HC/refine) regardless of
    which GPU runs it. Must match the implementation in invert_gpt2.py and
    refine_seeds.py so all phases attack the SAME fixed noisy target.
    """
    if sample_id is None:
        return torch.randn(dim, device=device, dtype=dtype)
    g = torch.Generator()
    g.manual_seed(NOISE_BASE_SEED + int(sample_id))
    return torch.randn(dim, generator=g).to(device=device, dtype=dtype)


def count_encoder_tokens(encoder, texts):
    """Count how many tokens `texts` would occupy in the encoder's tokenizer.

    Matches the counting convention used by SentenceTransformerWrapper and
    refine_seeds.py: sum of tokenizer.encode(t, add_special_tokens=True) lengths.
    """
    if isinstance(texts, str):
        texts = [texts]
    tokenizer = encoder.tokenizer
    return sum(len(tokenizer.encode(str(t), add_special_tokens=True)) for t in texts)


# =============================================================================
# VOCAB + TOKEN MAPPING SETUP (shared with invert_gpt2 pattern)
# =============================================================================

def build_vocab_embeddings(encoder, tokenizer, min_token_length=4, batch_size=512,
                           use_mask=True, word_initial_only=True):
    """Encode every LLM vocab token through the sentence encoder.

    Tokens shorter than min_token_length or containing disallowed characters
    get their embeddings zeroed out so they never rank high in similarity search.

    When word_initial_only=True (the default), only BPE tokens that decode
    with a leading space (i.e. whole-word tokens) are eligible. This prevents
    substitution artifacts like "andambleursday" where a subword-continuation
    token (e.g. "ambleursday") gets placed mid-sentence and concatenates
    without a space to its neighbor.

    Returns:
        vocab_embeddings: [vocab_size, hidden_dim] on CPU, normalized
        vocab_decoded: list of decoded (stripped) strings for debug printing
        vocab_valid: list of bools, True if token is eligible as a substitution
    """
    allowed_chars = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789 ,.\'-'
    )
    mask_prefix = "[MASK] [MASK] [MASK] [MASK] "

    vocab_size = tokenizer.vocab_size
    vocab_texts = []
    vocab_decoded = []
    vocab_valid = []
    n_word_initial = 0
    for i in range(vocab_size):
        # Decode WITHOUT stripping so we can detect whether this BPE token
        # represents a word-initial unit (leading-space convention in BPE).
        raw = tokenizer.decode([i])
        stripped = raw.strip()
        vocab_decoded.append(stripped)
        is_word_initial = raw.startswith(' ')
        if is_word_initial:
            n_word_initial += 1
        is_valid = (
            len(stripped) >= min_token_length
            and all(c in allowed_chars for c in stripped)
            and (not word_initial_only or is_word_initial)
        )
        vocab_valid.append(is_valid)
        if use_mask and stripped:
            vocab_texts.append(mask_prefix + stripped)
        else:
            vocab_texts.append(stripped if stripped else " ")

    all_embs = []
    for vi in tqdm(range(0, len(vocab_texts), batch_size), desc="Encoding vocab"):
        batch = vocab_texts[vi:vi + batch_size]
        embs = encoder.encode(
            batch, convert_to_tensor=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        all_embs.append(embs.cpu())
    vocab_embeddings = torch.cat(all_embs, dim=0)

    for i, valid in enumerate(vocab_valid):
        if not valid:
            vocab_embeddings[i] = 0.0

    print(f"    word_initial tokens in raw vocab: {n_word_initial}/{vocab_size}")
    return vocab_embeddings, vocab_decoded, vocab_valid


def build_similar_token_mapping(tokenizer, vocab_size, max_similar=8):
    """Group tokens by normalized form (case/whitespace variants).

    E.g., 'person', ' Person', 'PERSON' all map to the same group. Used to
    detect "already used" tokens for exclusion from the candidate pool.
    """
    mapping = torch.zeros((vocab_size, max_similar), dtype=torch.long)
    vocab = tokenizer.get_vocab()
    normalized_to_ids = {}

    for token_str, token_id in vocab.items():
        if token_id >= vocab_size:
            continue
        try:
            decoded = tokenizer.decode([token_id])
            normalized = decoded.strip().upper()
            if normalized:
                normalized_to_ids.setdefault(normalized, []).append(token_id)
        except Exception:
            pass

    for _, ids in normalized_to_ids.items():
        for token_id in ids:
            for i, similar_id in enumerate(ids[:max_similar]):
                mapping[token_id, i] = similar_id

    return mapping


# =============================================================================
# HILL-CLIMB CORE
# =============================================================================

def _compute_llm_candidates(llm_model, tokens, pool_size, vocab_valid_mask):
    with torch.no_grad():
        input_ids = torch.tensor([tokens], device=llm_model.device)
        out = llm_model(input_ids)
        logits = out.logits[0]
    llm_vocab = logits.size(-1)
    mask = vocab_valid_mask
    if mask.size(0) != llm_vocab:
        if mask.size(0) < llm_vocab:
            padded = torch.zeros(llm_vocab, dtype=torch.bool)
            padded[:mask.size(0)] = mask
            mask = padded
        else:
            mask = mask[:llm_vocab]
    k = min(pool_size, int(mask.sum().item()))
    if k <= 0:
        return [[] for _ in range(len(tokens))]
    invalid = ~mask.to(logits.device)
    per_pos = []
    for pos in range(len(tokens)):
        if pos == 0:
            per_pos.append([])
            continue
        pos_logits = logits[pos - 1].clone()
        pos_logits[invalid] = float("-inf")
        per_pos.append(pos_logits.topk(k).indices.tolist())
    return per_pos


def hill_climb(encoder, tokenizer, target_emb, seed_text,
               vocab_embeddings, similar_mapping,
               top_k=20, max_passes=3, exclude_used=True,
               min_pass_gain=0.004,
               emb_cache=None, verbose=True,
               use_saliency=False, saliency_skip_threshold=0.0,
               keep_frac=1.0, position_select='saliency', saliency_order='desc',
               probe_encoder=None, probe_target_emb=None, probe_keep_frac=0.25,
               llm_model=None, llm_pool_size=0,
               vocab_valid_mask=None):
    """Black-box local search: substitute tokens to maximize cosine similarity.

    Args:
        encoder: SentenceTransformer
        tokenizer: LLM tokenizer (defines vocab / token boundaries)
        target_emb: target embedding, shape [D] or [1, D], normalized
        seed_text: starting text
        vocab_embeddings: [vocab_size, D] with short/invalid tokens zeroed
        similar_mapping: [vocab_size, max_similar] case-variant groups
        top_k: candidates per position (from vocab top-K by sim to target)
        max_passes: max full sweeps over positions before giving up
        exclude_used: if True, drop candidates whose case-variant is already
                      in the seed (recomputed per pass)
        emb_cache: optional dict for per-sample encoding cache
        verbose: print per-pass progress

    Returns:
        (final_text, final_similarity) tuple
    """
    if emb_cache is None:
        emb_cache = {}

    t_emb = target_emb.detach().cpu()
    if t_emb.dim() == 2:
        t_emb = t_emb[0]
    t_emb_2d = t_emb.unsqueeze(0)

    tokens_spent = [0]

    def encode_cached(texts):
        missing = [s for s in texts if s not in emb_cache]
        seen = set()
        to_enc = [s for s in missing if not (s in seen or seen.add(s))]
        if to_enc:
            tokens_spent[0] += count_encoder_tokens(encoder, to_enc)
            embs = encoder.encode(
                to_enc, convert_to_tensor=True,
                normalize_embeddings=True, show_progress_bar=False,
            ).cpu()
            for j, s in enumerate(to_enc):
                emb_cache[s] = embs[j]
        return torch.stack([emb_cache[s] for s in texts])

    probe_emb_cache = {}
    probe_t_emb_2d = None
    if probe_encoder is not None and probe_target_emb is not None:
        p = probe_target_emb.detach().cpu()
        if p.dim() == 2:
            p = p[0]
        probe_t_emb_2d = p.unsqueeze(0)

    def probe_encode_cached(texts):
        if probe_encoder is None:
            return None
        missing = [s for s in texts if s not in probe_emb_cache]
        seen = set()
        to_enc = [s for s in missing if not (s in seen or seen.add(s))]
        if to_enc:
            embs = probe_encoder.encode(
                to_enc, convert_to_tensor=True,
                normalize_embeddings=True, show_progress_bar=False,
            ).cpu()
            for j, s in enumerate(to_enc):
                probe_emb_cache[s] = embs[j]
        return torch.stack([probe_emb_cache[s] for s in texts])

    pool_size = top_k * 8
    sims_all = torch.matmul(vocab_embeddings, t_emb)
    _, pool_ids = sims_all.topk(min(pool_size, int((sims_all > 0).sum().item())))
    candidate_pool = pool_ids.tolist()
    global_pool_set = set(candidate_pool)

    tokens = tokenizer.encode(seed_text, add_special_tokens=False)
    if not tokens:
        return seed_text, 0.0

    current_text = tokenizer.decode(tokens, skip_special_tokens=True)
    baseline_emb = encode_cached([current_text])[0]
    baseline_sim = F.cosine_similarity(t_emb_2d, baseline_emb.unsqueeze(0), dim=1).item()
    current_sim = baseline_sim

    llm_pool_per_pos = None
    if llm_model is not None and llm_pool_size > 0 and vocab_valid_mask is not None:
        llm_pool_per_pos = _compute_llm_candidates(
            llm_model, tokens, llm_pool_size, vocab_valid_mask,
        )

    saliency_drops = None
    position_order = list(range(len(tokens)))
    if use_saliency and len(tokens) > 1:
        sal_sentences = []
        for pos in range(len(tokens)):
            masked = tokens[:pos] + tokens[pos + 1:]
            sal_sentences.append(tokenizer.decode(masked, skip_special_tokens=True))
        sal_embs = encode_cached(sal_sentences)
        sal_sims = F.cosine_similarity(
            t_emb_2d.expand(sal_embs.size(0), -1), sal_embs, dim=1,
        )
        saliency_drops = (baseline_sim - sal_sims).tolist()
        sign = 1.0 if saliency_order == 'asc' else -1.0
        ranked = sorted(range(len(tokens)), key=lambda p: sign * saliency_drops[p])
        if saliency_skip_threshold > 0:
            position_order = [p for p in ranked if saliency_drops[p] >= saliency_skip_threshold]
        else:
            position_order = ranked

    if keep_frac < 1.0 and len(tokens) > 1:
        keep_n = max(1, int(round(keep_frac * len(tokens))))
        if position_select == 'random':
            rng = random.Random(hashlib.sha256(seed_text.encode('utf-8')).hexdigest())
            chosen = rng.sample(range(len(tokens)), min(keep_n, len(tokens)))
            position_order = [p for p in position_order if p in set(chosen)] or chosen
        elif position_select == 'first':
            chosen = set(range(min(keep_n, len(tokens))))
            position_order = [p for p in position_order if p in chosen] or sorted(chosen)
        else:
            position_order = position_order[:keep_n]
        if verbose:
            kept = len(position_order)
            print(f"  [hill-climb] saliency: ranked {len(tokens)} positions, "
                  f"kept {kept} (threshold={saliency_skip_threshold})")

    if verbose:
        extras = []
        if probe_encoder is not None:
            extras.append(f"probe(keep={probe_keep_frac:.2f})")
        if llm_pool_per_pos is not None:
            extras.append(f"llm_pool({llm_pool_size})")
        if use_saliency:
            extras.append("saliency")
        extras_str = (" " + " ".join(extras)) if extras else ""
        print(f"  [hill-climb] start: sim={baseline_sim:.4f}  tokens={len(tokens)}  "
              f"pool={len(candidate_pool)}  top_k={top_k}  exclude_used={exclude_used}{extras_str}")

    total_swaps = 0
    mapping_size = similar_mapping.size(0)

    for pass_idx in range(max_passes):
        if exclude_used:
            used = set()
            for t in tokens:
                if t < mapping_size:
                    used.update(similar_mapping[t].tolist())
                else:
                    used.add(t)
            used.discard(0)
            filtered_pool = [c for c in candidate_pool if c not in used]
            global_candidates_for_pass = filtered_pool[:top_k]
        else:
            global_candidates_for_pass = candidate_pool[:top_k]

        if not global_candidates_for_pass and llm_pool_per_pos is None:
            if verbose:
                print(f"  [hill-climb] pass {pass_idx+1}: empty candidate pool after exclusion")
            break

        pass_swaps = 0
        pass_gain = 0.0
        for pos in position_order:
            if llm_pool_per_pos is not None and pos < len(llm_pool_per_pos) and llm_pool_per_pos[pos]:
                cand_ids = [c for c in llm_pool_per_pos[pos] if c in global_pool_set] or llm_pool_per_pos[pos]
                cand_ids = cand_ids[:top_k]
            else:
                cand_ids = global_candidates_for_pass

            cand_sentences = []
            cand_tokenlists = []
            for cid in cand_ids:
                if cid == tokens[pos]:
                    continue
                new_toks = tokens[:]
                new_toks[pos] = cid
                new_text = tokenizer.decode(new_toks, skip_special_tokens=True)
                cand_sentences.append(new_text)
                cand_tokenlists.append(new_toks)

            if not cand_sentences:
                continue

            if probe_encoder is not None and probe_t_emb_2d is not None and len(cand_sentences) > 1:
                keep_m = max(1, int(round(len(cand_sentences) * probe_keep_frac)))
                if keep_m < len(cand_sentences):
                    probe_embs = probe_encode_cached(cand_sentences)
                    probe_sims = F.cosine_similarity(
                        probe_t_emb_2d.expand(probe_embs.size(0), -1), probe_embs, dim=1,
                    )
                    top_idx = probe_sims.topk(keep_m).indices.tolist()
                    cand_sentences = [cand_sentences[i] for i in top_idx]
                    cand_tokenlists = [cand_tokenlists[i] for i in top_idx]

            embs = encode_cached(cand_sentences)
            sims_batch = F.cosine_similarity(
                t_emb_2d.expand(embs.size(0), -1), embs, dim=1,
            )
            best_idx = int(sims_batch.argmax().item())
            best_sim = sims_batch[best_idx].item()

            if best_sim > current_sim:
                pass_gain += (best_sim - current_sim)
                tokens = cand_tokenlists[best_idx]
                current_sim = best_sim
                pass_swaps += 1

        total_swaps += pass_swaps
        if verbose:
            print(f"  [hill-climb] pass {pass_idx+1}: swaps={pass_swaps}  "
                  f"sim={current_sim:.4f}  Δ={pass_gain:+.4f}  pool_after_excl={len(global_candidates_for_pass)}")

        if pass_swaps == 0:
            if verbose:
                print(f"  [hill-climb] converged after {pass_idx+1} passes (no swaps)")
            break

        # Early stop: if this pass's total gain is below the threshold, remaining
        # passes are unlikely to improve things meaningfully.
        if pass_gain < min_pass_gain:
            if verbose:
                print(f"  [hill-climb] early-stop after pass {pass_idx+1}: "
                      f"Δ={pass_gain:.4f} < min_pass_gain={min_pass_gain}")
            break

    final_text = tokenizer.decode(tokens, skip_special_tokens=True)
    if verbose:
        print(f"  [hill-climb] encoder tokens spent inside hill-climb: {tokens_spent[0]}")
    return final_text, current_sim, tokens_spent[0]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hill-climb post-processing on seed reconstructions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--input', type=str, required=True,
                        help="Input CSV with seed texts")
    parser.add_argument('--output', type=str, required=True,
                        help="Output CSV (will include new hill_climbed_* columns)")
    parser.add_argument('--column', type=str, required=True,
                        help="Column containing seed text (e.g., c1, refined_text)")
    parser.add_argument('--target-column', type=str, default='source_text',
                        help="Column containing target/ground-truth text (for target embedding)")

    parser.add_argument('--encoder', type=str, default='contriever',
                        help=f"Encoder: {list(ENCODERS.keys())} or full HF path")
    parser.add_argument('--llm', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help="LLM whose vocab is used for candidate pool + tokenization")
    parser.add_argument('--trust-remote-code', action='store_true',
                        help="Trust remote code for encoder/tokenizer")

    parser.add_argument('--top-k', type=int, default=20,
                        help="Candidates per position (after used-token filtering)")
    parser.add_argument('--max-passes', type=int, default=3,
                        help="Max full sweeps over token positions")
    parser.add_argument('--min-pass-gain', type=float, default=0.004,
                        help="Early-stop if a pass's total similarity gain is below this. "
                             "Set to 0 to only stop on full convergence (no swaps).")
    parser.add_argument('--min-token-length', type=int, default=4,
                        help="Min decoded char length for a token to be in candidate pool")
    parser.add_argument('--no-exclude-used', action='store_true',
                        help="Don't exclude tokens whose case-variants are already in the seed")
    parser.add_argument('--allow-subword-candidates', action='store_true',
                        help="Allow non-word-initial BPE subwords as substitution candidates. "
                             "Default: only word-initial tokens (prevents 'andambleursday' artifacts).")

    parser.add_argument('--saliency', action='store_true',
                        help="Compute per-position saliency (mask-then-encode) and visit "
                             "positions in descending impact order.")
    parser.add_argument('--saliency-order', type=str, default='desc',
                        choices=['desc', 'asc'],
                        help="Visit positions by decreasing (desc) or increasing (asc) occlusion drop.")
    parser.add_argument('--keep-frac', type=float, default=1.0,
                        help="Fraction of positions to optimise. 1.0 = all.")
    parser.add_argument('--position-select', type=str, default='saliency',
                        choices=['saliency', 'random', 'first'],
                        help="Which positions --keep-frac keeps: most salient, random, or leftmost.")
    parser.add_argument('--saliency-skip-threshold', type=float, default=0.0,
                        help="Skip positions whose saliency drop is below this. 0 = visit all.")

    parser.add_argument('--probe-encoder', type=str, default=None,
                        help="Cheap encoder to pre-score candidates per position before "
                             "the real encoder is queried. Example: 'sentence-transformers/all-MiniLM-L6-v2'.")
    parser.add_argument('--probe-keep-frac', type=float, default=0.25,
                        help="Fraction of candidates per position to forward to the real encoder.")

    parser.add_argument('--llm-pool-size', type=int, default=0,
                        help="If > 0, build per-position candidate pools from the LLM's "
                             "distribution at each position (top-K LLM-likely tokens, "
                             "intersected with the global vocab-similarity pool).")

    parser.add_argument('--resume', action='store_true',
                        help="If --output exists, load it and skip rows where the "
                             "output text column is already populated.")
    parser.add_argument('--noise-level', type=float, default=0.0,
                        help="Gaussian sigma added to the target embedding (defense "
                             "simulation, matches the ZSInvert noise threat model).")
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--skip-samples', type=int, default=0)
    parser.add_argument('--checkpoint-interval', type=int, default=10)

    # Output column names default to '<input_column>_hc', '<input_column>_hc_sim', etc.
    # This means running hill_climb on multiple source columns (c1, refined_text, ...)
    # in the same CSV does not overwrite prior results.
    parser.add_argument('--output-text-column', type=str, default=None,
                        help="Column for hill-climbed text (default: '{column}_hc')")
    parser.add_argument('--output-sim-column', type=str, default=None,
                        help="Column for hill-climbed similarity (default: '{column}_hc_sim')")
    parser.add_argument('--output-tokens-column', type=str, default=None,
                        help="Column for per-sample encoder token count "
                             "(default: '{column}_hc_tokens')")
    parser.add_argument('--output-seed-sim-column', type=str, default=None,
                        help="Column for re-computed seed similarity "
                             "(default: '{column}_hc_seed_sim'). Distinct from "
                             "'{column}_sim' so Phase 1 scores are preserved.")
    parser.add_argument('--device', type=str, default=None,
                        help="Device (default: auto-detect)")

    args = parser.parse_args()

    from run_metadata import track_experiment
    track_experiment(args.output, args)

    # Resolve column-derived defaults so outputs don't collide across multiple
    # HC runs on the same CSV (e.g., one on 'c1' and another on 'refined_text').
    if args.output_text_column is None:
        args.output_text_column = f"{args.column}_hc"
    if args.output_sim_column is None:
        args.output_sim_column = f"{args.column}_hc_sim"
    if args.output_tokens_column is None:
        args.output_tokens_column = f"{args.column}_hc_tokens"
    if args.output_seed_sim_column is None:
        args.output_seed_sim_column = f"{args.column}_hc_seed_sim"

    print(f"[Output columns] text='{args.output_text_column}'  "
          f"sim='{args.output_sim_column}'  "
          f"tokens='{args.output_tokens_column}'  "
          f"seed_sim='{args.output_seed_sim_column}'")

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    encoder_name = ENCODERS.get(args.encoder, args.encoder)
    print(f"[Loading encoder] {encoder_name}")
    trust_remote = args.trust_remote_code or 'qwen' in encoder_name.lower()
    encoder = SentenceTransformer(encoder_name, device=device, trust_remote_code=trust_remote)

    print(f"[Loading tokenizer] {args.llm}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)

    vocab_size = tokenizer.vocab_size
    word_initial_only = not args.allow_subword_candidates
    print(f"[Pre-computing vocab embeddings]  vocab={vocab_size}  "
          f"min_len={args.min_token_length}  word_initial_only={word_initial_only}")
    vocab_embeddings, _, vocab_valid = build_vocab_embeddings(
        encoder, tokenizer,
        min_token_length=args.min_token_length,
        word_initial_only=word_initial_only,
    )
    print(f"  {sum(vocab_valid)}/{vocab_size} valid tokens eligible for substitution")

    print(f"[Building similar-token mapping]")
    similar_mapping = build_similar_token_mapping(tokenizer, vocab_size)

    vocab_valid_mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i, v in enumerate(vocab_valid):
        if v:
            vocab_valid_mask[i] = True

    probe_encoder = None
    if args.probe_encoder:
        print(f"[Loading probe encoder] {args.probe_encoder}")
        probe_encoder = SentenceTransformer(args.probe_encoder, device=device)

    llm_model = None
    if args.llm_pool_size > 0:
        from transformers import AutoModelForCausalLM
        print(f"[Loading LLM for per-position pool] {args.llm}")
        llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm, trust_remote_code=True, torch_dtype=torch.float16,
        ).to(device).eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resumed_count = 0
    if args.resume and out_path.exists():
        print(f"[Resume] Loading existing output: {out_path}")
        df = pd.read_csv(out_path)
        print(f"  {len(df)} rows")
        if args.output_text_column in df.columns:
            resumed_count = int(df[args.output_text_column].notna().sum())
            print(f"  {resumed_count} rows already processed (will be skipped)")
    else:
        print(f"[Loading input] {args.input}")
        df = pd.read_csv(args.input)
        print(f"  {len(df)} rows")

    if args.output_text_column not in df.columns:
        df[args.output_text_column] = None
    if args.output_sim_column not in df.columns:
        df[args.output_sim_column] = None
    if args.output_tokens_column not in df.columns:
        df[args.output_tokens_column] = None
    if args.output_seed_sim_column not in df.columns:
        df[args.output_seed_sim_column] = None

    start_idx = args.skip_samples
    end_idx = min(len(df), start_idx + args.max_samples) if args.max_samples else len(df)

    total_baseline = 0.0
    total_final = 0.0
    total_tokens_spent = 0
    n_done = 0
    n_improved = 0

    for idx in tqdm(range(start_idx, end_idx), desc="Hill-climbing"):
        if args.resume and pd.notna(df.at[idx, args.output_text_column]):
            continue
        seed = df.at[idx, args.column]
        target = df.at[idx, args.target_column]

        if pd.isna(seed) or pd.isna(target):
            continue
        seed = str(seed).strip()
        target = str(target).strip()
        if not seed or not target:
            continue

        sample_tokens = 0
        sample_tokens += count_encoder_tokens(encoder, [target])
        with torch.no_grad():
            target_emb = encoder.encode(
                [target], convert_to_tensor=True, normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        if args.noise_level > 0:
            sample_id = df.at[idx, 'id'] if 'id' in df.columns else idx
            noise = _deterministic_noise(target_emb.shape[-1], sample_id, target_emb.device, target_emb.dtype)
            target_emb = target_emb + noise * args.noise_level
            target_emb = F.normalize(target_emb, p=2, dim=-1)

        emb_cache = {}

        sample_tokens += count_encoder_tokens(encoder, [seed])
        with torch.no_grad():
            seed_emb = encoder.encode(
                [seed], convert_to_tensor=True, normalize_embeddings=True,
                show_progress_bar=False,
            )[0].cpu()
        emb_cache[seed] = seed_emb
        seed_sim = F.cosine_similarity(
            target_emb.cpu().unsqueeze(0), seed_emb.unsqueeze(0), dim=1
        ).item()

        probe_target_emb = None
        if probe_encoder is not None:
            with torch.no_grad():
                probe_target_emb = probe_encoder.encode(
                    [target], convert_to_tensor=True, normalize_embeddings=True,
                    show_progress_bar=False,
                )[0].cpu()

        print(f"\n[Sample {idx}]")
        print(f"  Seed:  sim={seed_sim:.4f}  | {seed[:90]}")

        final_text, final_sim, hc_tokens = hill_climb(
            encoder=encoder, tokenizer=tokenizer, target_emb=target_emb,
            seed_text=seed, vocab_embeddings=vocab_embeddings,
            similar_mapping=similar_mapping,
            top_k=args.top_k, max_passes=args.max_passes,
            exclude_used=not args.no_exclude_used,
            min_pass_gain=args.min_pass_gain,
            emb_cache=emb_cache, verbose=True,
            use_saliency=args.saliency,
            saliency_skip_threshold=args.saliency_skip_threshold,
            keep_frac=args.keep_frac,
            position_select=args.position_select,
            saliency_order=args.saliency_order,
            probe_encoder=probe_encoder,
            probe_target_emb=probe_target_emb,
            probe_keep_frac=args.probe_keep_frac,
            llm_model=llm_model,
            llm_pool_size=args.llm_pool_size,
            vocab_valid_mask=vocab_valid_mask,
        )
        sample_tokens += hc_tokens

        delta = final_sim - seed_sim
        marker = "↑" if delta > 0 else ("=" if delta == 0 else "↓")
        print(f"  Final: sim={final_sim:.4f} {marker} Δ={delta:+.4f}  "
              f"tokens={sample_tokens}  | {final_text[:90]}")

        df.at[idx, args.output_text_column] = final_text
        df.at[idx, args.output_sim_column] = final_sim
        df.at[idx, args.output_tokens_column] = sample_tokens
        df.at[idx, args.output_seed_sim_column] = seed_sim

        total_baseline += seed_sim
        total_final += final_sim
        total_tokens_spent += sample_tokens
        n_done += 1
        if final_sim > seed_sim:
            n_improved += 1

        if n_done % args.checkpoint_interval == 0:
            df.to_csv(out_path, index=False)

    df.to_csv(out_path, index=False)
    print(f"\n[Saved] {out_path}")

    if n_done > 0:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Samples processed:         {n_done}")
        print(f"Improved:                  {n_improved}/{n_done} ({100.0*n_improved/n_done:.1f}%)")
        print(f"Mean seed similarity:      {total_baseline/n_done:.4f}")
        print(f"Mean final similarity:     {total_final/n_done:.4f}")
        print(f"Mean delta:                {(total_final - total_baseline)/n_done:+.4f}")
        print(f"Mean tokens/sample:        {total_tokens_spent/n_done:.0f}")
        print(f"Total encoder tokens:      {total_tokens_spent}  "
              f"(not counting one-time vocab pre-encoding)")


if __name__ == '__main__':
    main()