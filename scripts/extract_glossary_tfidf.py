"""Auto-extract domain glossary from EPIC Spanish transcripts via TF-IDF."""
import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="Dir of Spanish .txt transcripts")
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    docs = [p.read_text(encoding="utf-8") for p in sorted(Path(args.corpus).glob("*.txt"))]

    vec = TfidfVectorizer(
        ngram_range=(1, 3),
        max_df=0.85,
        min_df=2,
        sublinear_tf=True,
    )
    X = vec.fit_transform(docs)
    scores = np.asarray(X.mean(axis=0)).flatten()
    terms_scores = sorted(zip(vec.get_feature_names_out(), scores),
                          key=lambda x: x[1], reverse=True)

    # Output as {es_term: ""} — translations filled in later (manually or LLM)
    glossary = {term: "" for term, _ in terms_scores[: args.top_k]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(glossary, indent=2, ensure_ascii=False))
    print(f"Extracted {len(glossary)} terms → {args.output}")


if __name__ == "__main__":
    main()
