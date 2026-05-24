from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FIELD_RE = re.compile(r"^(Title|Source|Link|Published Date|Author|Tags|Summary|Content):\s*(.*)$")
TOKEN_RE = re.compile(
    r"[0-9A-Za-zÀ-ỹĐđ]+(?:[-_][0-9A-Za-zÀ-ỹĐđ]+)?",
    flags=re.UNICODE,
)

METHODS = ("lead", "lexrank", "position_lexrank", "position_lexrank_mmr")
METHOD_LABELS = {
    "lead": "Lead-k",
    "lexrank": "Vanilla LexRank",
    "position_lexrank": "Position-Aware LexRank",
    "position_lexrank_mmr": "Position-Aware LexRank + MMR",
}

VIETNAMESE_STOPWORDS = {
    "a",
    "ai",
    "anh",
    "ay",
    "ấy",
    "bị",
    "bởi",
    "cả",
    "các",
    "cái",
    "cần",
    "càng",
    "chỉ",
    "cho",
    "chưa",
    "chúng",
    "có",
    "của",
    "cùng",
    "cũng",
    "đã",
    "đang",
    "đây",
    "để",
    "đến",
    "đi",
    "đó",
    "được",
    "dù",
    "hay",
    "hơn",
    "khi",
    "không",
    "là",
    "lại",
    "lên",
    "lúc",
    "mà",
    "một",
    "này",
    "nên",
    "nếu",
    "ngay",
    "người",
    "như",
    "nhưng",
    "những",
    "nơi",
    "nữa",
    "ở",
    "phải",
    "qua",
    "ra",
    "rằng",
    "rất",
    "rồi",
    "sau",
    "sẽ",
    "so",
    "sự",
    "tại",
    "theo",
    "thì",
    "trên",
    "trong",
    "trước",
    "từ",
    "từng",
    "và",
    "vào",
    "vẫn",
    "về",
    "vì",
    "với",
}


@dataclass(frozen=True)
class Article:
    path: Path
    title: str
    content: str
    reference_summary: str = ""


@dataclass(frozen=True)
class ArticleGroup:
    group_id: str
    path: Path
    articles: list[Article]
    reference_summaries: list[str] | None = None

    @property
    def title(self) -> str:
        titles = [article.title for article in self.articles if article.title]
        return " ".join(titles)

    @property
    def reference_summary(self) -> str:
        if self.reference_summaries:
            return " ".join(self.reference_summaries)
        summaries = [article.reference_summary for article in self.articles if article.reference_summary]
        return " ".join(summaries)

    @property
    def references(self) -> list[str]:
        if self.reference_summaries:
            return self.reference_summaries
        fallback = self.reference_summary
        return [fallback] if fallback else []


@dataclass(frozen=True)
class SentenceRecord:
    text: str
    article_index: int
    local_index: int
    source_path: Path


@dataclass(frozen=True)
class SummaryResult:
    source: Path
    title: str
    sentences: list[str]
    selected_indices: list[int]
    relevance_scores: list[float]
    method: str = "position_lexrank_mmr"
    redundancy: float = 0.0
    selected_sources: list[str] | None = None

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def fix_mojibake(text: str) -> str:
    """Repair common UTF-8-read-as-Latin-1 mojibake without touching valid text."""
    suspicious = ("Ã", "Ä", "á»", "áº", "Â", "â€")
    if not any(marker in text for marker in suspicious):
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    original_badness = sum(text.count(marker) for marker in suspicious)
    repaired_badness = sum(repaired.count(marker) for marker in suspicious)
    return repaired if repaired_badness < original_badness else text


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            return fix_mojibake(path.read_text(encoding=encoding))
        except UnicodeError:
            continue
    return fix_mojibake(path.read_text(errors="replace"))


def parse_article(path: Path) -> Article:
    text = read_text(path).replace("\ufeff", "")
    fields: dict[str, list[str]] = {}
    current_field: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = FIELD_RE.match(line)
        if match:
            current_field = match.group(1)
            fields.setdefault(current_field, []).append(match.group(2).strip())
            continue
        if current_field:
            fields.setdefault(current_field, []).append(line)

    title = " ".join(part for part in fields.get("Title", []) if part).strip()
    reference = " ".join(part for part in fields.get("Summary", []) if part).strip()
    content = "\n".join(part for part in fields.get("Content", []) if part).strip()
    if not content:
        content = text.strip()
    return Article(path=path, title=title, content=content, reference_summary=reference)


def parse_article_group(
    group_id: str,
    group_path: Path,
    files: list[Path],
    reference_files: list[Path] | None = None,
) -> ArticleGroup:
    articles = [parse_article(path) for path in sorted(files)]
    reference_summaries = None
    if reference_files:
        reference_summaries = []
        for path in sorted(reference_files):
            text = read_text(path).strip()
            if text:
                reference_summaries.append(text)
    return ArticleGroup(
        group_id=group_id,
        path=group_path,
        articles=articles,
        reference_summaries=reference_summaries,
    )


def is_list_like_sentence(sentence: str) -> bool:
    token_count = len(tokenize(sentence, keep_stopwords=True))
    comma_count = sentence.count(",")
    parenthesis_count = sentence.count("(") + sentence.count(")")
    if token_count > 60:
        return True
    if comma_count >= 8:
        return True
    if parenthesis_count >= 6 and comma_count >= 3:
        return True
    return bool(
        re.match(
            r"^(danh sách|thủ môn|hậu vệ|tiền vệ|tiền đạo)\b.*:",
            sentence,
            flags=re.IGNORECASE,
        )
    )


def split_sentences(
    text: str,
    min_tokens: int = 4,
    drop_list_sentences: bool = True,
) -> list[str]:
    sentences: list[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = re.sub(r"\s+", " ", paragraph.strip())
        if not paragraph:
            continue
        parts = re.split(r"(?<=[.!?…])\s+(?=[\"'“”‘’(\[]?[A-ZÀ-ỸĐ0-9])", paragraph)
        for part in parts:
            sentence = part.strip(" \t-–—")
            if len(tokenize(sentence, keep_stopwords=True)) >= min_tokens:
                if drop_list_sentences and is_list_like_sentence(sentence):
                    continue
                sentences.append(sentence)
    return sentences


def normalize_token(token: str) -> str:
    return token.lower().strip("_-")


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    tokens = [normalize_token(token) for token in TOKEN_RE.findall(text)]
    tokens = [token for token in tokens if token]
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in VIETNAMESE_STOPWORDS and len(token) > 1]


def build_tfidf_vectors(tokenized_sentences: list[list[str]]) -> list[dict[str, float]]:
    n_docs = len(tokenized_sentences)
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_sentences:
        doc_freq.update(set(tokens))

    vectors: list[dict[str, float]] = []
    for tokens in tokenized_sentences:
        counts = Counter(tokens)
        token_count = max(len(tokens), 1)
        vector: dict[str, float] = {}
        for token, count in counts.items():
            tf = count / token_count
            idf = math.log((1 + n_docs) / (1 + doc_freq[token])) + 1.0
            vector[token] = tf * idf
        vectors.append(vector)
    return vectors


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    numerator = sum(value * vec_b.get(term, 0.0) for term, value in vec_a.items())
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


def build_similarity_matrix(vectors: list[dict[str, float]]) -> list[list[float]]:
    size = len(vectors)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            sim = cosine_similarity(vectors[i], vectors[j])
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix


def normalize_distribution(values: Iterable[float], size: int) -> list[float]:
    values = [max(0.0, value) for value in values]
    total = sum(values)
    if total <= 0.0:
        return [1.0 / size] * size
    return [value / total for value in values]


def build_position_prior(
    sentences: list[str],
    vectors: list[dict[str, float]],
    title: str = "",
    position_weight: float = 0.8,
) -> list[float]:
    size = len(sentences)
    position_values = [1.0 / ((index + 1) ** 0.85) for index in range(size)]
    position_prior = normalize_distribution(position_values, size)

    title_tokens = tokenize(title)
    if not title_tokens:
        return position_prior

    title_vector = build_tfidf_vectors([title_tokens] + [tokenize(sentence) for sentence in sentences])[0]
    title_values = [cosine_similarity(title_vector, vector) for vector in vectors]
    title_prior = normalize_distribution(title_values, size)

    position_weight = min(max(position_weight, 0.0), 1.0)
    return normalize_distribution(
        (
            position_weight * position_prior[index]
            + (1.0 - position_weight) * title_prior[index]
            for index in range(size)
        ),
        size,
    )


def lexrank(
    similarity_matrix: list[list[float]],
    prior: list[float],
    threshold: float = 0.1,
    damping: float = 0.85,
    max_iter: int = 100,
    tolerance: float = 1e-6,
) -> list[float]:
    size = len(similarity_matrix)
    if size == 0:
        return []

    transition = [[0.0] * size for _ in range(size)]
    for row_idx, row in enumerate(similarity_matrix):
        weights = [value if value >= threshold else 0.0 for value in row]
        row_sum = sum(weights)
        if row_sum <= 0.0:
            transition[row_idx] = prior[:]
        else:
            transition[row_idx] = [value / row_sum for value in weights]

    scores = [1.0 / size] * size
    for _ in range(max_iter):
        next_scores = [(1.0 - damping) * prior[index] for index in range(size)]
        for source in range(size):
            source_score = damping * scores[source]
            for target in range(size):
                next_scores[target] += source_score * transition[source][target]
        diff = sum(abs(next_scores[index] - scores[index]) for index in range(size))
        scores = next_scores
        if diff < tolerance:
            break
    return normalize_distribution(scores, size)


def mmr_select(
    relevance_scores: list[float],
    similarity_matrix: list[list[float]],
    max_sentences: int,
    lambda_mmr: float = 0.7,
) -> list[int]:
    size = len(relevance_scores)
    if size == 0:
        return []

    max_sentences = min(max_sentences, size)
    lambda_mmr = min(max(lambda_mmr, 0.0), 1.0)
    max_relevance = max(relevance_scores) if relevance_scores else 0.0
    if max_relevance > 0.0:
        scaled_relevance = [score / max_relevance for score in relevance_scores]
    else:
        scaled_relevance = [0.0] * size
    selected: list[int] = []
    candidates = set(range(size))

    while candidates and len(selected) < max_sentences:
        best_index = None
        best_score = -float("inf")
        for candidate in candidates:
            redundancy = 0.0
            if selected:
                redundancy = max(similarity_matrix[candidate][picked] for picked in selected)
            score = lambda_mmr * scaled_relevance[candidate] - (1.0 - lambda_mmr) * redundancy
            if score > best_score:
                best_score = score
                best_index = candidate
        if best_index is None:
            break
        selected.append(best_index)
        candidates.remove(best_index)

    return selected


def top_relevance_select(relevance_scores: list[float], max_sentences: int) -> list[int]:
    return sorted(
        range(len(relevance_scores)),
        key=lambda index: (-relevance_scores[index], index),
    )[:max_sentences]


def average_pairwise_similarity(
    selected_indices: list[int],
    similarity_matrix: list[list[float]],
) -> float:
    if len(selected_indices) < 2:
        return 0.0

    total = 0.0
    pair_count = 0
    for left_pos, left_idx in enumerate(selected_indices):
        for right_idx in selected_indices[left_pos + 1 :]:
            total += similarity_matrix[left_idx][right_idx]
            pair_count += 1
    return total / pair_count if pair_count else 0.0


def build_position_prior_for_records(
    records: list[SentenceRecord],
    vectors: list[dict[str, float]],
    title: str = "",
    position_weight: float = 0.8,
) -> list[float]:
    size = len(records)
    position_values = [1.0 / ((record.local_index + 1) ** 0.85) for record in records]
    position_prior = normalize_distribution(position_values, size)

    title_tokens = tokenize(title)
    if not title_tokens:
        return position_prior

    title_vector = build_tfidf_vectors([title_tokens] + [tokenize(record.text) for record in records])[0]
    title_values = [cosine_similarity(title_vector, vector) for vector in vectors]
    title_prior = normalize_distribution(title_values, size)

    position_weight = min(max(position_weight, 0.0), 1.0)
    return normalize_distribution(
        (
            position_weight * position_prior[index]
            + (1.0 - position_weight) * title_prior[index]
            for index in range(size)
        ),
        size,
    )


def summarize_sentence_records(
    source: Path,
    title: str,
    records: list[SentenceRecord],
    prior_title: str | None = None,
    method: str = "position_lexrank_mmr",
    max_sentences: int = 3,
    ratio: float | None = None,
    threshold: float = 0.1,
    damping: float = 0.85,
    position_weight: float = 0.8,
    lambda_mmr: float = 0.7,
    min_sentence_tokens: int = 4,
    drop_list_sentences: bool = True,
) -> SummaryResult:
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose one of: {', '.join(METHODS)}")

    if not records:
        return SummaryResult(source, title, [], [], [], method=method)

    sentences = [record.text for record in records]

    if ratio is not None:
        max_sentences = max(1, math.ceil(len(sentences) * ratio))
    max_sentences = min(max_sentences, len(sentences))

    tokenized = [tokenize(sentence) for sentence in sentences]
    vectors = build_tfidf_vectors(tokenized)
    similarity_matrix = build_similarity_matrix(vectors)

    if method == "lead":
        relevance_scores = [0.0] * len(sentences)
        selected = list(range(max_sentences))
    else:
        if method == "lexrank":
            prior = [1.0 / len(sentences)] * len(sentences)
        else:
            prior = build_position_prior_for_records(
                records,
                vectors,
                title=prior_title or title,
                position_weight=position_weight,
            )

        relevance_scores = lexrank(
            similarity_matrix,
            prior=prior,
            threshold=threshold,
            damping=damping,
        )
        if method == "position_lexrank_mmr":
            selected = mmr_select(
                relevance_scores,
                similarity_matrix,
                max_sentences=max_sentences,
                lambda_mmr=lambda_mmr,
            )
        else:
            selected = top_relevance_select(relevance_scores, max_sentences=max_sentences)

    selected_in_original_order = sorted(selected)
    redundancy = average_pairwise_similarity(selected_in_original_order, similarity_matrix)
    return SummaryResult(
        source=source,
        title=title,
        sentences=[sentences[index] for index in selected_in_original_order],
        selected_indices=selected_in_original_order,
        relevance_scores=relevance_scores,
        method=method,
        redundancy=redundancy,
        selected_sources=[str(records[index].source_path) for index in selected_in_original_order],
    )


def summarize_article(
    article: Article,
    method: str = "position_lexrank_mmr",
    max_sentences: int = 3,
    ratio: float | None = None,
    threshold: float = 0.1,
    damping: float = 0.85,
    position_weight: float = 0.8,
    lambda_mmr: float = 0.7,
    min_sentence_tokens: int = 4,
    drop_list_sentences: bool = True,
) -> SummaryResult:
    sentences = split_sentences(
        article.content,
        min_tokens=min_sentence_tokens,
        drop_list_sentences=drop_list_sentences,
    )
    records = [
        SentenceRecord(
            text=sentence,
            article_index=0,
            local_index=index,
            source_path=article.path,
        )
        for index, sentence in enumerate(sentences)
    ]
    return summarize_sentence_records(
        source=article.path,
        title=article.title,
        records=records,
        prior_title=article.title,
        method=method,
        max_sentences=max_sentences,
        ratio=ratio,
        threshold=threshold,
        damping=damping,
        position_weight=position_weight,
        lambda_mmr=lambda_mmr,
        min_sentence_tokens=min_sentence_tokens,
        drop_list_sentences=drop_list_sentences,
    )


def summarize_group(
    group: ArticleGroup,
    method: str = "position_lexrank_mmr",
    max_sentences: int = 5,
    ratio: float | None = None,
    threshold: float = 0.1,
    damping: float = 0.85,
    position_weight: float = 0.8,
    lambda_mmr: float = 0.7,
    min_sentence_tokens: int = 4,
    drop_list_sentences: bool = True,
) -> SummaryResult:
    records: list[SentenceRecord] = []
    for article_index, article in enumerate(group.articles):
        sentences = split_sentences(
            article.content,
            min_tokens=min_sentence_tokens,
            drop_list_sentences=drop_list_sentences,
        )
        records.extend(
            SentenceRecord(
                text=sentence,
                article_index=article_index,
                local_index=sentence_index,
                source_path=article.path,
            )
            for sentence_index, sentence in enumerate(sentences)
        )

    return summarize_sentence_records(
        source=group.path,
        title=group.group_id,
        records=records,
        prior_title=group.title,
        method=method,
        max_sentences=max_sentences,
        ratio=ratio,
        threshold=threshold,
        damping=damping,
        position_weight=position_weight,
        lambda_mmr=lambda_mmr,
        min_sentence_tokens=min_sentence_tokens,
        drop_list_sentences=drop_list_sentences,
    )


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def rouge_n(candidate: str, reference: str, n: int) -> dict[str, float]:
    cand_counts = Counter(ngrams(tokenize(candidate, keep_stopwords=True), n))
    ref_counts = Counter(ngrams(tokenize(reference, keep_stopwords=True), n))
    overlap = sum((cand_counts & ref_counts).values())
    cand_total = sum(cand_counts.values())
    ref_total = sum(ref_counts.values())
    precision = overlap / cand_total if cand_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str) -> dict[str, float]:
    candidate_tokens = tokenize(candidate, keep_stopwords=True)
    reference_tokens = tokenize(reference, keep_stopwords=True)
    overlap = lcs_length(candidate_tokens, reference_tokens)
    precision = overlap / len(candidate_tokens) if candidate_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.txt"))


def collect_reference_files(reference_dir: Path) -> list[Path]:
    if not reference_dir.is_dir():
        return []
    gold_files = sorted(reference_dir.glob("*.gold.txt"))
    if gold_files:
        return gold_files
    return sorted(
        path
        for path in reference_dir.glob("*.txt")
        if not path.name.startswith(".")
    )


def collect_article_groups(input_path: Path) -> list[ArticleGroup]:
    if input_path.is_file():
        return [parse_article_group(input_path.stem, input_path.parent, [input_path])]

    original_root = input_path / "original"
    summary_root = input_path / "summary"
    if original_root.is_dir() and summary_root.is_dir():
        groups: list[ArticleGroup] = []
        for cluster_dir in sorted(original_root.iterdir()):
            source_dir = cluster_dir / "original"
            if not source_dir.is_dir():
                continue
            files = sorted(source_dir.glob("*.txt"))
            reference_dir = summary_root / cluster_dir.name
            reference_files = collect_reference_files(reference_dir)
            if files:
                groups.append(
                    parse_article_group(
                        cluster_dir.name,
                        cluster_dir,
                        files,
                        reference_files=reference_files,
                    )
                )
        return groups

    cluster_dirs = [
        child
        for child in sorted(input_path.iterdir())
        if child.is_dir() and (child / "original").is_dir()
    ]
    if cluster_dirs:
        groups: list[ArticleGroup] = []
        for cluster_dir in cluster_dirs:
            files = sorted((cluster_dir / "original").glob("*.txt"))
            if files:
                groups.append(parse_article_group(cluster_dir.name, cluster_dir, files))
        return groups

    direct_files = sorted(input_path.glob("*.txt"))
    if direct_files:
        return [
            parse_article_group(path.stem, path.parent, [path])
            for path in direct_files
        ]

    recursive_files = sorted(input_path.rglob("*.txt"))
    return [
        parse_article_group(path.stem, path.parent, [path])
        for path in recursive_files
    ]


def write_summary(output_dir: Path, result: SummaryResult) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if result.method == "position_lexrank_mmr":
        output_path = output_dir / f"{result.source.stem}_summary.txt"
    else:
        output_path = output_dir / f"{result.source.stem}_{result.method}_summary.txt"
    lines = []
    if result.title:
        lines.append(f"Title: {result.title}")
        lines.append("")
    lines.append(result.text)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path


def average_metric(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        key: sum(row[key] for row in metric_rows) / len(metric_rows)
        for key in ("precision", "recall", "f1")
    }


def print_metric(name: str, values: dict[str, float]) -> None:
    print(
        f"{name}: P={values['precision']:.4f} "
        f"R={values['recall']:.4f} F1={values['f1']:.4f}"
    )


def evaluate_summary(candidate: str, reference: str) -> dict[str, float]:
    r1 = rouge_n(candidate, reference, 1)
    r2 = rouge_n(candidate, reference, 2)
    rl = rouge_l(candidate, reference)
    return {
        "rouge1_p": r1["precision"],
        "rouge1_r": r1["recall"],
        "rouge1_f1": r1["f1"],
        "rouge2_p": r2["precision"],
        "rouge2_r": r2["recall"],
        "rouge2_f1": r2["f1"],
        "rougel_p": rl["precision"],
        "rougel_r": rl["recall"],
        "rougel_f1": rl["f1"],
    }


def evaluate_against_references(candidate: str, references: list[str]) -> dict[str, float]:
    if not references:
        return evaluate_summary(candidate, "")
    metric_rows = [evaluate_summary(candidate, reference) for reference in references]
    keys = metric_rows[0].keys()
    return {
        key: sum(row[key] for row in metric_rows) / len(metric_rows)
        for key in keys
    }


def run_experiments(
    groups: list[ArticleGroup] | list[Path],
    methods: Iterable[str] = METHODS,
    max_sentences: int = 5,
    ratio: float | None = None,
    threshold: float = 0.1,
    damping: float = 0.85,
    position_weight: float = 0.8,
    lambda_mmr: float = 0.7,
    min_sentence_tokens: int = 4,
    drop_list_sentences: bool = True,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for item in groups:
        if isinstance(item, ArticleGroup):
            group = item
        else:
            group = parse_article_group(item.stem, item.parent, [item])
        references = group.references
        if not references:
            continue
        for method in methods:
            result = summarize_group(
                group,
                method=method,
                max_sentences=max_sentences,
                ratio=ratio,
                threshold=threshold,
                damping=damping,
                position_weight=position_weight,
                lambda_mmr=lambda_mmr,
                min_sentence_tokens=min_sentence_tokens,
                drop_list_sentences=drop_list_sentences,
            )
            metrics = evaluate_against_references(result.text, references)
            selected_source_count = len(set(result.selected_sources or []))
            source_coverage = selected_source_count / len(group.articles) if group.articles else 0.0
            rows.append(
                {
                    "file": group.group_id,
                    "documents": float(len(group.articles)),
                    "references": float(len(references)),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "sentences": float(len(result.sentences)),
                    "selected_sources": float(selected_source_count),
                    "source_coverage": source_coverage,
                    "redundancy": result.redundancy,
                    **metrics,
                }
            )
    return rows


def write_group_summaries(
    groups: Iterable[ArticleGroup] | Iterable[Path],
    output_dir: Path,
    method: str = "position_lexrank_mmr",
    max_sentences: int = 5,
    ratio: float | None = None,
    threshold: float = 0.1,
    damping: float = 0.85,
    position_weight: float = 0.8,
    lambda_mmr: float = 0.7,
    min_sentence_tokens: int = 4,
    drop_list_sentences: bool = True,
) -> list[Path]:
    output_paths: list[Path] = []
    for item in groups:
        if isinstance(item, ArticleGroup):
            group = item
        else:
            group = parse_article_group(item.stem, item.parent, [item])
        result = summarize_group(
            group,
            method=method,
            max_sentences=max_sentences,
            ratio=ratio,
            threshold=threshold,
            damping=damping,
            position_weight=position_weight,
            lambda_mmr=lambda_mmr,
            min_sentence_tokens=min_sentence_tokens,
            drop_list_sentences=drop_list_sentences,
        )
        output_paths.append(write_summary(output_dir, result))
    return output_paths


def average_experiment_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    method_order = {method: index for index, method in enumerate(METHODS)}
    grouped: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    averages: list[dict[str, float | str]] = []
    metric_keys = (
        "rouge1_f1",
        "rouge2_f1",
        "rougel_f1",
        "redundancy",
        "source_coverage",
    )
    for method, method_rows in grouped.items():
        average_row: dict[str, float | str] = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
        }
        for key in metric_keys:
            average_row[key] = sum(float(row[key]) for row in method_rows) / len(method_rows)
        averages.append(average_row)

    return sorted(averages, key=lambda row: method_order.get(str(row["method"]), 999))


def print_experiment_table(rows: list[dict[str, float | str]]) -> None:
    headers = ("Method", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "Redundancy", "SrcCover")
    print(
        f"{headers[0]:<32} {headers[1]:>11} {headers[2]:>11} "
        f"{headers[3]:>11} {headers[4]:>11} {headers[5]:>9}"
    )
    print("-" * 94)
    for row in rows:
        print(
            f"{str(row['method_label']):<32} "
            f"{float(row['rouge1_f1']):>11.4f} "
            f"{float(row['rouge2_f1']):>11.4f} "
            f"{float(row['rougel_f1']):>11.4f} "
            f"{float(row['redundancy']):>11.4f} "
            f"{float(row.get('source_coverage', 0.0)):>9.4f}"
        )


def write_rows_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vietnamese extractive news summarization with Position-Aware LexRank and MMR.",
    )
    parser.add_argument("--input", type=Path, default=Path("data"), help="A .txt article or a folder of .txt files.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Folder for generated summaries.")
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="position_lexrank_mmr",
        help="Summarization method to run.",
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Run Lead-k, LexRank, Position-Aware LexRank, and Position-Aware LexRank + MMR.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path for detailed comparison rows. Also writes *_average.csv.",
    )
    parser.add_argument(
        "--save-summaries",
        action="store_true",
        help="When comparing methods, also write summaries for --method to --output.",
    )
    parser.add_argument("--max-sentences", type=int, default=5, help="Maximum sentences in each summary.")
    parser.add_argument("--ratio", type=float, default=None, help="Optional summary length ratio, e.g. 0.25.")
    parser.add_argument("--threshold", type=float, default=0.1, help="LexRank cosine edge threshold.")
    parser.add_argument("--damping", type=float, default=0.85, help="PageRank damping factor.")
    parser.add_argument(
        "--position-weight",
        type=float,
        default=0.8,
        help="Weight for early-sentence prior; the rest comes from title similarity.",
    )
    parser.add_argument(
        "--lambda-mmr",
        type=float,
        default=0.7,
        help="MMR relevance/diversity trade-off. Higher means more LexRank relevance.",
    )
    parser.add_argument("--min-sentence-tokens", type=int, default=4, help="Drop very short sentences.")
    parser.add_argument(
        "--keep-list-sentences",
        action="store_true",
        help="Keep roster/table-like list sentences instead of filtering them.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate with ROUGE. If data/summary exists, uses *.gold.txt references; otherwise falls back to Summary fields.",
    )
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()
    groups = collect_article_groups(args.input)
    if not groups:
        print(f"No article groups found in {args.input}", file=sys.stderr)
        return 1

    if args.compare_methods:
        rows = run_experiments(
            groups,
            max_sentences=args.max_sentences,
            ratio=args.ratio,
            threshold=args.threshold,
            damping=args.damping,
            position_weight=args.position_weight,
            lambda_mmr=args.lambda_mmr,
            min_sentence_tokens=args.min_sentence_tokens,
            drop_list_sentences=not args.keep_list_sentences,
        )
        if not rows:
            print("No reference summaries found for comparison.", file=sys.stderr)
            return 1
        average_rows = average_experiment_rows(rows)
        print_experiment_table(average_rows)
        if args.csv:
            write_rows_csv(args.csv, rows)
            average_path = args.csv.with_name(f"{args.csv.stem}_average{args.csv.suffix}")
            write_rows_csv(average_path, average_rows)
            print(f"\nWrote detailed rows to {args.csv}")
            print(f"Wrote average rows to {average_path}")
        if args.save_summaries:
            output_paths = write_group_summaries(
                groups,
                args.output,
                method=args.method,
                max_sentences=args.max_sentences,
                ratio=args.ratio,
                threshold=args.threshold,
                damping=args.damping,
                position_weight=args.position_weight,
                lambda_mmr=args.lambda_mmr,
                min_sentence_tokens=args.min_sentence_tokens,
                drop_list_sentences=not args.keep_list_sentences,
            )
            print(
                f"Wrote {len(output_paths)} {METHOD_LABELS[args.method]} "
                f"summaries to {args.output}"
            )
        return 0

    rouge_1_rows: list[dict[str, float]] = []
    rouge_2_rows: list[dict[str, float]] = []
    rouge_l_rows: list[dict[str, float]] = []
    redundancy_rows: list[float] = []

    for group in groups:
        result = summarize_group(
            group,
            method=args.method,
            max_sentences=args.max_sentences,
            ratio=args.ratio,
            threshold=args.threshold,
            damping=args.damping,
            position_weight=args.position_weight,
            lambda_mmr=args.lambda_mmr,
            min_sentence_tokens=args.min_sentence_tokens,
            drop_list_sentences=not args.keep_list_sentences,
        )
        output_path = write_summary(args.output, result)
        print(f"[{group.group_id} | {len(group.articles)} documents] -> {output_path}")
        print(result.text)
        print()

        references = group.references
        if args.evaluate and references and result.text:
            metrics = evaluate_against_references(result.text, references)
            r1 = {
                "precision": metrics["rouge1_p"],
                "recall": metrics["rouge1_r"],
                "f1": metrics["rouge1_f1"],
            }
            r2 = {
                "precision": metrics["rouge2_p"],
                "recall": metrics["rouge2_r"],
                "f1": metrics["rouge2_f1"],
            }
            rl = {
                "precision": metrics["rougel_p"],
                "recall": metrics["rougel_r"],
                "f1": metrics["rougel_f1"],
            }
            rouge_1_rows.append(r1)
            rouge_2_rows.append(r2)
            rouge_l_rows.append(rl)
            redundancy_rows.append(result.redundancy)
            print(f"  References: {len(references)}")
            if result.selected_sources:
                print(f"  Selected sources: {len(set(result.selected_sources))}/{len(group.articles)}")
            print_metric("  ROUGE-1", r1)
            print_metric("  ROUGE-2", r2)
            print_metric("  ROUGE-L", rl)
            print(f"  Redundancy: {result.redundancy:.4f}")
            print()

    if args.evaluate and rouge_1_rows:
        print("Average")
        print_metric("  ROUGE-1", average_metric(rouge_1_rows))
        print_metric("  ROUGE-2", average_metric(rouge_2_rows))
        print_metric("  ROUGE-L", average_metric(rouge_l_rows))
        print(f"  Redundancy: {sum(redundancy_rows) / len(redundancy_rows):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
