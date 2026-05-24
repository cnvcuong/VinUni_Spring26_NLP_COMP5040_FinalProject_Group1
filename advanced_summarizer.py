"""
advanced_summarizer.py
======================
Advanced Vietnamese extractive summarisation — BERT-based methods.

Methods
-------
bert_centroid
    Pure BERT: score each sentence by cosine similarity to the cluster
    centroid (mean of all sentence embeddings). No graph, no PageRank.
pacsum
    PACSUM directed-graph centrality (Zheng & Lapata, ACL 2019).
    Edges from earlier sentences carry full weight; edges from later
    sentences are penalised by beta, creating a position-aware ranking.

Both methods use sentence-transformers embeddings
(paraphrase-multilingual-MiniLM-L12-v2 by default).

Performance tip
---------------
Call build_embeddings_cache(groups, embedder) ONCE before any experiments.
Pass the cache to run_experiments_bert and every grid-search call so that
BERT encoding happens only once per cluster, not once per config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

import position_aware_lexrank_mmr as base

# ── Method registry ───────────────────────────────────────────────────────────

METHODS_BERT = (
    "bert_centroid",
    "pacsum",
)

METHOD_LABELS_BERT: dict[str, str] = {
    "bert_centroid": "BERT Centroid",
    "pacsum":        "PACSUM",
}

# ── Sentence embedder ─────────────────────────────────────────────────────────

class SentenceEmbedder:
    """
    Thin wrapper around a sentence-transformers model.
    Lazy-loads on first encode() call.
    Automatically uses CUDA if available; warns and falls back to CPU.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print(
                "[SentenceEmbedder] WARNING: CUDA not available — running on CPU (slow).\n"
                "  Fix: pip install torch --index-url https://download.pytorch.org/whl/cu126",
                flush=True,
            )
        else:
            print(f"[SentenceEmbedder] Using GPU: {torch.cuda.get_device_name(0)}", flush=True)

        print(f"[SentenceEmbedder] Loading '{self.model_name}' ...", flush=True)
        self._model = SentenceTransformer(self.model_name, device=device)
        print("[SentenceEmbedder] Model ready.", flush=True)

    def encode(self, sentences: list[str]) -> np.ndarray:
        """Return (N, D) float32 array of L2-normalised embeddings."""
        if self._model is None:
            self._load()
        return self._model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
        )


# ── Per-cluster embedding cache ───────────────────────────────────────────────

@dataclass
class ClusterCache:
    """
    Pre-computed BERT data for one article cluster.
    Avoids re-encoding sentences for every method / hyperparameter config.
    """
    group_id:        str
    records:         list[base.SentenceRecord]
    embeddings:      np.ndarray           # (N, D), L2-normalised
    sim_matrix:      list[list[float]]    # N x N cosine sim, diagonal = 0
    title_embedding: np.ndarray | None    # (D,) or None


def _build_records(
    group: base.ArticleGroup,
    min_sentence_tokens: int = 4,
    drop_list_sentences: bool = True,
) -> list[base.SentenceRecord]:
    records: list[base.SentenceRecord] = []
    for art_idx, article in enumerate(group.articles):
        sentences = base.split_sentences(
            article.content,
            min_tokens=min_sentence_tokens,
            drop_list_sentences=drop_list_sentences,
        )
        records.extend(
            base.SentenceRecord(
                text=sent,
                article_index=art_idx,
                local_index=sent_idx,
                source_path=article.path,
            )
            for sent_idx, sent in enumerate(sentences)
        )
    return records


def _cosine_sim_matrix(embeddings: np.ndarray) -> list[list[float]]:
    """N x N cosine similarity matrix; diagonal forced to 0."""
    raw = (embeddings @ embeddings.T).astype(float)
    n = raw.shape[0]
    sim: list[list[float]] = []
    for i in range(n):
        row = [max(0.0, min(1.0, float(raw[i, j]))) for j in range(n)]
        row[i] = 0.0
        sim.append(row)
    return sim


def build_cluster_cache(
    group: base.ArticleGroup,
    embedder: SentenceEmbedder,
    min_sentence_tokens: int = 4,
    drop_list_sentences: bool = True,
) -> ClusterCache:
    """Encode all sentences in one cluster and return a reusable cache."""
    records = _build_records(group, min_sentence_tokens, drop_list_sentences)

    if not records:
        return ClusterCache(
            group_id=group.group_id,
            records=[],
            embeddings=np.zeros((0, 1), dtype=np.float32),
            sim_matrix=[],
            title_embedding=None,
        )

    sentences_text = [r.text for r in records]
    embeddings = embedder.encode(sentences_text)
    sim_matrix = _cosine_sim_matrix(embeddings)

    title_text = group.title
    title_embedding: np.ndarray | None = None
    if title_text.strip():
        title_embedding = embedder.encode([title_text])[0]

    return ClusterCache(
        group_id=group.group_id,
        records=records,
        embeddings=embeddings,
        sim_matrix=sim_matrix,
        title_embedding=title_embedding,
    )


def build_embeddings_cache(
    groups: list[base.ArticleGroup],
    embedder: SentenceEmbedder,
    min_sentence_tokens: int = 4,
    drop_list_sentences: bool = True,
) -> dict[str, ClusterCache]:
    """
    Pre-compute BERT embeddings for every cluster.
    Call ONCE before all experiments — the cache is reused everywhere.
    """
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(groups, desc="Building embedding cache", unit="cluster")
    except ImportError:
        iterator = groups
        print(f"Building embedding cache for {len(groups)} clusters ...", flush=True)

    cache: dict[str, ClusterCache] = {}
    for group in iterator:
        cache[group.group_id] = build_cluster_cache(
            group, embedder, min_sentence_tokens, drop_list_sentences
        )
    return cache


# ── Scoring functions ─────────────────────────────────────────────────────────

def bert_centroid_scores(embeddings: np.ndarray) -> list[float]:
    """
    Pure BERT centroid scoring — no graph, no PageRank.

    Steps:
    1. Compute cluster centroid = mean of all L2-normalised sentence embeddings.
    2. Re-normalise the centroid.
    3. Score each sentence = cosine similarity to the centroid.

    Sentences that best represent the overall topic of the cluster rank highest.
    """
    n = len(embeddings)
    if n == 0:
        return []
    centroid = embeddings.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    scores = [max(0.0, float(embeddings[i] @ centroid)) for i in range(n)]
    return base.normalize_distribution(scores, n)


def pacsum_scores(
    sim_matrix: list[list[float]],
    beta: float = 0.0,
) -> list[float]:
    """
    PACSUM centrality (Zheng & Lapata, ACL 2019).

    Score(i) = sum_{j < i} sim(j, i)          # earlier -> i  (full weight)
             + beta * sum_{j > i} sim(j, i)    # later   -> i  (penalised)

    beta=0: only forward edges count (maximum position bias).
    beta=1: symmetric, same as plain degree centrality.
    """
    n = len(sim_matrix)
    scores = [0.0] * n
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            scores[i] += sim_matrix[i][j] if j < i else beta * sim_matrix[i][j]
    return base.normalize_distribution(scores, n)


# ── Core summarisation ────────────────────────────────────────────────────────

def _summarize_from_cache(
    group: base.ArticleGroup,
    cache: ClusterCache,
    method: str,
    max_sentences: int = 5,
    pacsum_beta: float = 0.0,
) -> base.SummaryResult:
    """Run one method using pre-computed embeddings — no encoding here."""
    records = cache.records
    embeddings = cache.embeddings
    sim_matrix = cache.sim_matrix

    if not records:
        return base.SummaryResult(group.path, group.group_id, [], [], [], method=method)

    sentences_text = [r.text for r in records]
    max_sentences = min(max_sentences, len(sentences_text))

    # ── Score sentences ───────────────────────────────────────────────────────
    if method == "bert_centroid":
        relevance_scores = bert_centroid_scores(embeddings)

    elif method == "pacsum":
        relevance_scores = pacsum_scores(sim_matrix, beta=pacsum_beta)

    else:
        relevance_scores = [1.0 / len(sentences_text)] * len(sentences_text)

    # ── Select top-k sentences (in document order) ────────────────────────────
    selected = base.top_relevance_select(relevance_scores, max_sentences)
    selected_ordered = sorted(selected)

    # Redundancy: average BERT pairwise cosine sim among selected sentences
    bert_sim = (embeddings @ embeddings.T).astype(float)
    bert_sim_lists = [
        [float(bert_sim[i, j]) for j in range(len(sentences_text))]
        for i in range(len(sentences_text))
    ]
    redundancy = base.average_pairwise_similarity(selected_ordered, bert_sim_lists)

    return base.SummaryResult(
        source=group.path,
        title=group.group_id,
        sentences=[sentences_text[i] for i in selected_ordered],
        selected_indices=selected_ordered,
        relevance_scores=relevance_scores,
        method=method,
        redundancy=redundancy,
        selected_sources=[str(records[i].source_path) for i in selected_ordered],
    )


def summarize_group_bert(
    group: base.ArticleGroup,
    embedder: SentenceEmbedder,
    method: str = "bert_centroid",
    max_sentences: int = 5,
    pacsum_beta: float = 0.0,
    min_sentence_tokens: int = 4,
    drop_list_sentences: bool = True,
    cluster_cache: ClusterCache | None = None,
) -> base.SummaryResult:
    """
    Summarise one cluster using BERT-based methods.
    Pass cluster_cache to avoid re-encoding (critical for grid search).
    """
    if method not in METHODS_BERT:
        raise ValueError(f"Unknown method '{method}'. Choose from: {METHODS_BERT}")

    if cluster_cache is None:
        cluster_cache = build_cluster_cache(
            group, embedder, min_sentence_tokens, drop_list_sentences
        )

    return _summarize_from_cache(
        group, cluster_cache, method,
        max_sentences=max_sentences,
        pacsum_beta=pacsum_beta,
    )


# ── BERTScore evaluation ──────────────────────────────────────────────────────

def evaluate_with_bertscore(
    candidates: list[str],
    references: list[str],
    lang: str = "vi",
    model_type: str = "bert-base-multilingual-cased",
    verbose: bool = False,
) -> dict[str, float]:
    """BERTScore P/R/F1 averaged over (candidate, reference) pairs."""
    from bert_score import score as _bert_score
    P, R, F1 = _bert_score(
        candidates,
        references,
        lang=lang,
        model_type=model_type,
        verbose=verbose,
    )
    return {
        "bertscore_p":  float(P.mean()),
        "bertscore_r":  float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiments_bert(
    groups: list[base.ArticleGroup],
    embedder: SentenceEmbedder,
    methods: Iterable[str] = METHODS_BERT,
    max_sentences: int = 5,
    pacsum_beta: float = 0.0,
    min_sentence_tokens: int = 4,
    drop_list_sentences: bool = True,
    compute_bertscore: bool = False,
    embeddings_cache: dict[str, ClusterCache] | None = None,
) -> list[dict]:
    """
    Run BERT methods across all clusters.
    Pass embeddings_cache to avoid re-encoding — essential for grid search.
    """
    rows: list[dict] = []
    methods_list = list(methods)

    for group in groups:
        references = group.references
        if not references:
            continue

        cache = (
            embeddings_cache.get(group.group_id)
            if embeddings_cache
            else build_cluster_cache(group, embedder, min_sentence_tokens, drop_list_sentences)
        )

        for method in methods_list:
            result = _summarize_from_cache(
                group, cache, method,
                max_sentences=max_sentences,
                pacsum_beta=pacsum_beta,
            )
            metrics = base.evaluate_against_references(result.text, references)
            selected_src_count = len(set(result.selected_sources or []))
            source_coverage = (
                selected_src_count / len(group.articles) if group.articles else 0.0
            )

            row: dict = {
                "file":             group.group_id,
                "documents":        float(len(group.articles)),
                "references":       float(len(references)),
                "method":           method,
                "method_label":     METHOD_LABELS_BERT[method],
                "sentences":        float(len(result.sentences)),
                "selected_sources": float(selected_src_count),
                "source_coverage":  source_coverage,
                "redundancy":       result.redundancy,
                **metrics,
            }

            if compute_bertscore and result.text and group.reference_summary:
                row.update(evaluate_with_bertscore(
                    [result.text], [group.reference_summary]
                ))

            rows.append(row)

    return rows


def average_experiment_rows_bert(
    rows: list[dict],
    methods=METHODS_BERT,
) -> list[dict]:
    """Average per-cluster rows by method, preserving method order."""
    method_order = {m: i for i, m in enumerate(methods)}
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    base_keys = ["rouge1_f1", "rouge2_f1", "rougel_f1", "redundancy", "source_coverage"]
    sample_row = next(iter(grouped.values()), [{}])[0]
    extra_keys = [k for k in ("bertscore_p", "bertscore_r", "bertscore_f1") if k in sample_row]
    metric_keys = base_keys + extra_keys

    averages: list[dict] = []
    for method, method_rows in grouped.items():
        avg_row: dict = {
            "method":       method,
            "method_label": METHOD_LABELS_BERT.get(method, method),
        }
        for key in metric_keys:
            if key in method_rows[0]:
                avg_row[key] = sum(float(r[key]) for r in method_rows) / len(method_rows)
        averages.append(avg_row)

    return sorted(averages, key=lambda r: method_order.get(str(r["method"]), 999))
