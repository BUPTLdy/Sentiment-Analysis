import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import sentiment_cli


HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


class CleanReviewsTest(unittest.TestCase):
    def test_clean_reviews_removes_empty_values_and_normalizes_spaces(self):
        self.assertEqual(
            sentiment_cli.clean_reviews([None, "  很好  ", "\u3000一般\u3000", "", "nan"]),
            ["很好", "一般"],
        )

    def test_parser_requires_a_subcommand(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sentiment_cli.build_parser().parse_args([])


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is not installed")
class PipelineTest(unittest.TestCase):
    def test_pipeline_can_fit_and_predict_small_chinese_dataset(self):
        positive = ["质量很好", "非常满意", "物流很快", "值得推荐"]
        negative = ["质量太差", "非常失望", "物流很慢", "不会再买"]
        pipeline = sentiment_cli.build_pipeline(min_df=1, max_features=1_000)
        pipeline.fit(positive + negative, [1] * len(positive) + [0] * len(negative))

        predictions = pipeline.predict(["质量很好", "质量太差"])

        self.assertEqual(predictions.tolist(), [1, 0])

    def test_model_round_trip_validates_format(self):
        import joblib

        pipeline = sentiment_cli.build_pipeline(min_df=1, max_features=1_000)
        pipeline.fit(["非常好", "真的喜欢", "非常差", "真的讨厌"], [1, 1, 0, 0])

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.joblib"
            joblib.dump(
                {
                    "format_version": sentiment_cli.MODEL_FORMAT_VERSION,
                    "pipeline": pipeline,
                    "metadata": {"test": True},
                },
                model_path,
            )
            predictions = sentiment_cli.predict(model_path, ["非常好", "非常差"])

        self.assertEqual([item["label"] for item in predictions], ["positive", "negative"])


if __name__ == "__main__":
    unittest.main()
