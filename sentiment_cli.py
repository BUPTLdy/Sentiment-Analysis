#!/usr/bin/env python3
"""Train and run a modern Chinese sentiment-classification baseline."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MODEL_FORMAT_VERSION = 1
DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "sentiment_svm.joblib"


def clean_reviews(values: Iterable[object]) -> list[str]:
    """Convert spreadsheet values to non-empty, normalized strings."""

    reviews: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).replace("\u3000", " ").strip()
        if text and text.lower() != "nan":
            reviews.append(text)
    return reviews


def load_dataset(data_directory: Path) -> tuple[list[str], list[int]]:
    """Load positive and negative reviews from the repository XLS files."""

    import pandas as pd

    positive_path = data_directory / "pos.xls"
    negative_path = data_directory / "neg.xls"
    missing = [str(path) for path in (positive_path, negative_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing dataset file(s): {', '.join(missing)}")

    positive = clean_reviews(
        pd.read_excel(positive_path, header=None, engine="xlrd").iloc[:, 0]
    )
    negative = clean_reviews(
        pd.read_excel(negative_path, header=None, engine="xlrd").iloc[:, 0]
    )
    if not positive or not negative:
        raise ValueError("Both positive and negative datasets must contain reviews.")

    reviews = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)
    return reviews, labels


def build_pipeline(*, min_df: int = 2, max_features: int = 150_000):
    """Build a deterministic character n-gram sentiment pipeline."""

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LinearSVC(dual="auto", random_state=42)),
        ]
    )


def train_model(
    *,
    data_directory: Path,
    model_path: Path,
    test_size: float,
    random_state: int,
) -> dict[str, object]:
    """Train, evaluate, and persist a model plus reproducibility metadata."""

    import joblib
    import sklearn
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    reviews, labels = load_dataset(data_directory)
    train_reviews, test_reviews, train_labels, test_labels = train_test_split(
        reviews,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    pipeline = build_pipeline()
    pipeline.fit(train_reviews, train_labels)
    predicted_labels = pipeline.predict(test_reviews)
    accuracy = float(accuracy_score(test_labels, predicted_labels))

    metadata: dict[str, object] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "total_reviews": len(reviews),
        "training_reviews": len(train_reviews),
        "test_reviews": len(test_reviews),
        "test_accuracy": round(accuracy, 6),
        "test_size": test_size,
        "random_state": random_state,
        "labels": {"0": "negative", "1": "positive"},
        "features": "character TF-IDF n-grams (2-4)",
        "classifier": "LinearSVC",
    }
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "pipeline": pipeline,
        "metadata": metadata,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path)
    return {"model_path": str(model_path.resolve()), **metadata}


def load_model(model_path: Path) -> dict[str, object]:
    """Load a model produced by this CLI and validate its format."""

    import joblib

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run `python sentiment_cli.py train` first."
        )

    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or payload.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported model format. Retrain with the current CLI.")
    if "pipeline" not in payload or "metadata" not in payload:
        raise ValueError("Model payload is incomplete. Retrain with the current CLI.")
    return payload


def predict(model_path: Path, texts: list[str]) -> list[dict[str, object]]:
    """Predict labels and decision scores for input texts."""

    payload = load_model(model_path)
    pipeline = payload["pipeline"]
    labels = pipeline.predict(texts)
    scores = pipeline.decision_function(texts)
    return [
        {
            "text": text,
            "label": "positive" if int(label) == 1 else "negative",
            "score": round(float(score), 6),
        }
        for text, label, score in zip(texts, labels, scores)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run a Chinese sentiment-classification baseline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate a model.")
    train_parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY, help="XLS data directory."
    )
    train_parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH, help="Output model path."
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)

    predict_parser = subparsers.add_parser("predict", help="Predict one or more reviews.")
    predict_parser.add_argument("texts", nargs="+", help="Review text to classify.")
    predict_parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH, help="Trained model path."
    )
    predict_parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON output."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            if not 0 < args.test_size < 1:
                raise ValueError("--test-size must be between 0 and 1.")
            result = train_model(
                data_directory=args.data_dir,
                model_path=args.model,
                test_size=args.test_size,
                random_state=args.random_state,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        results = predict(args.model, args.texts)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for result in results:
                print(f"{result['label']}\t{result['score']:+.6f}\t{result['text']}")
        return 0
    except (FileNotFoundError, ValueError) as error:
        build_parser().error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
