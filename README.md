# Vietnamese Extractive News Summarization

> **COMP5040 – Natural Language Processing | VinUniversity Spring 2026**
> Group 1 — Tran Trung Duc, Le Anh Thu, Luu Duc Toan, Nguyen Van Cuong

This project studies and implements an **extractive summarization** system for
Vietnamese news in a **multi-document** setting: each cluster groups articles
about the same event collected from multiple news sources.

The proposed method is **Position-Aware LexRank + MMR** — graph-based sentence
ranking (LexRank) augmented with a position-and-title-aware teleport prior and
Maximum Marginal Relevance (MMR) selection to reduce redundancy in the output
summary.

**Live demo:** https://nlp-final-project.streamlit.app
**Application source code:** https://github.com/Tointech/NLP-Project
**Project repository (report + code + data):** https://github.com/cnvcuong/VinUni_Spring26_NLP_COMP5040_FinalProject_Group1

---

## Key Results (300 clusters, ViMs dataset)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Redundancy |
|---|---|---|---|---|---|
| Lead-k | 0.4513 | 0.2731 | 0.2897 | 0.7363 | 0.0945 |
| Vanilla LexRank | 0.4656 | 0.2840 | 0.2862 | 0.7366 | 0.3307 |
| Position-Aware LexRank | 0.4690 | 0.2977 | 0.2922 | 0.7452 | 0.3369 |
| BERT Centroid | 0.4474 | 0.2609 | 0.2656 | 0.7314 | 0.6933 |
| PACSUM | 0.4524 | 0.2637 | 0.2705 | 0.7299 | 0.5176 |
| **Pos-Aware LexRank + MMR** ★ | **0.5135** | **0.3354** | **0.3044** | **0.7569** | 0.2147 |

★ Proposed method. Best grid-search configuration (λ=0.6, τ=0.05, α=0.7): ROUGE-L = **0.3140**.

---

## Repository Structure

```
.
├── position_aware_lexrank_mmr.py   # Core pipeline: TF-IDF, LexRank, position prior, MMR
├── advanced_summarizer.py          # BERT Centroid, PACSUM (used in experiments.ipynb)
├── experiments.ipynb               # Main experiment notebook (6 methods)
├── data/
│   ├── original/
│   │   └── Cluster_xxx/
│   │       └── original/
│   │           └── *.txt           # Source news articles
│   ├── summary/
│   │   └── Cluster_xxx/
│   │       ├── 0.gold.txt          # Reference summary 1
│   │       └── 1.gold.txt          # Reference summary 2
│   └── README.md                   # Dataset description (ViMs)
├── outputs/                        # Generated summaries and CSV files (auto-created)

```

Dataset: **ViMs** — https://github.com/CLC-HCMUS/ViMs-Dataset

---

## Installation

```bash
pip install rouge-score bert-score sentence-transformers torch numpy
```

Requires Python 3.9+. A CUDA-capable GPU is recommended for BERT-based methods
but not required — all methods run on CPU, with longer encoding time for BERT.

---

## Quick Start (proposed method)

```bash
python position_aware_lexrank_mmr.py --input data --output outputs --max-sentences 5
```

Summaries are written to `outputs/Cluster_xxx_summary.txt`.

---

## Run with ROUGE Evaluation

```bash
python position_aware_lexrank_mmr.py --input data --output outputs \
    --max-sentences 5 --evaluate
```

The script uses `data/summary/Cluster_xxx/*.gold.txt` as reference summaries.

---

## Compare All TF-IDF Methods

```bash
python position_aware_lexrank_mmr.py --input data --compare-methods --max-sentences 5
```

Export results to CSV:

```bash
python position_aware_lexrank_mmr.py --input data --compare-methods \
    --max-sentences 5 --csv outputs/results.csv --save-summaries
```

This compares four TF-IDF methods: `Lead-k`, `Vanilla LexRank`,
`Position-Aware LexRank`, and `Position-Aware LexRank + MMR`.

---

## Full Experiment Notebook

Open `experiments.ipynb` to run all six methods, the ablation study, grid
search, BERTScore evaluation, and view sample summaries:

```bash
jupyter notebook experiments.ipynb
```

The notebook covers:
- Comparison of all 6 methods: Lead-k, Vanilla LexRank, Position-Aware LexRank,
  **Position-Aware LexRank + MMR**, BERT Centroid, PACSUM
- Grid search over 80 configurations (λ × τ × α) for the proposed method
- PACSUM beta parameter sweep
- BERTScore evaluation with `bert-base-multilingual-cased`
- Sample output summaries for each method

> Building the BERT embedding cache takes approximately 2–3 minutes on an
> RTX 4060 Laptop GPU, or 20–30 minutes on CPU.

---

## Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `--method` | `lead` / `lexrank` / `position_lexrank` / `position_lexrank_mmr` | `position_lexrank_mmr` |
| `--max-sentences` | Maximum number of sentences in the summary | `5` |
| `--threshold` | Cosine similarity threshold for LexRank graph edges (τ) | `0.1` |
| `--position-weight` | Weight of positional decay in the teleport prior (α); remainder goes to title similarity | `0.8` |
| `--lambda-mmr` | Relevance–diversity trade-off in MMR (λ) | `0.7` |
| `--ratio` | Use a sentence ratio instead of a fixed count (e.g. `0.25`) | — |
| `--keep-list-sentences` | Keep list-like sentences (filtered by default) | `False` |

**Best configuration from grid search:** `--threshold 0.05 --position-weight 0.7 --lambda-mmr 0.6`

---

## MMR Formula

```
MMR(c) = λ × LexRank(c) − (1 − λ) × max_sim(c, s),   s ∈ S
```

- `S`: set of already-selected sentences
- `LexRank(c)`: centrality score of candidate sentence `c`, scaled to [0, 1]
- `max_sim(c, s)`: highest cosine similarity between `c` and any sentence in `S`
- λ close to `1.0`: prioritizes sentences with high LexRank scores
- λ close to `0.0`: penalizes redundancy more strongly

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **ROUGE-1/2/L F1** | Lexical overlap (unigram / bigram / LCS) with reference summaries |
| **BERTScore F1** | Semantic overlap using `bert-base-multilingual-cased` soft token alignment |
| **Redundancy** | Average pairwise cosine similarity among selected sentences — lower is better |
| **SrcCover** | Fraction of source articles contributing at least one selected sentence — higher means broader coverage |

---

## Streamlit Application

The interactive web application allows users to:
- Select a news article from pre-loaded categories (Economy, Technology, Sports,
  Environment, Health, Education)
- Upload a `.txt` file or paste Vietnamese text directly
- Choose a summarization method and adjust parameters via sidebar sliders
- View the extracted summary with per-sentence relevance scores

**Run locally:**

```bash
cd NLP-Project
pip install streamlit
streamlit run app.py
```

---

## Team

| Member | Student ID | Role |
|---|---|---|
| Tran Trung Duc | V202401788 | Proposed pipeline, TF-IDF experiments, first report draft |
| Le Anh Thu | V202503040 | Deployed application, comparative experiment design |
| Luu Duc Toan | V202502963 | BERT Centroid, PACSUM, BERTScore evaluation |
| Nguyen Van Cuong | V202502961 | Result consolidation, final report, repository management |
