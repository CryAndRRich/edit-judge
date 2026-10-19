import unittest

from PIL import ImageStat

from tests.helpers import make_dataset  # sets the temporary data folders first

from src.common.dataset import iter_images
from src.common.io import read_jsonl, write_jsonl
from src.filters.base import Detector, run, score_path
from src.filters.evaluate import best_f1, prf
from src.filters.voting import combine
from src.paths import FILTERS_DIR


class RedDetector(Detector):
    """Scores an image by its mean red value: a stand-in for a real model."""
    task, name, model_id, batch_size = "nsfw", "red", "test", 2

    def load(self, device):
        self.loaded = device

    def score(self, images):
        return [ImageStat.Stat(im).mean[0] / 255 for im in images]


class FilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        make_dataset(n_samples=3, n_edits=2)

    def test_images(self):
        ids = [r.image_id for r in iter_images()]
        self.assertEqual(len(ids), 9)
        self.assertIn("s001/02", ids)
        self.assertEqual(len(list(iter_images("originals"))), 3)

    def test_run_is_resumable(self):
        first = run(RedDetector(), limit=4, device="cpu")
        second = run(RedDetector(), device="cpu")
        self.assertEqual((first["scored"], second["scored"], second["already_done"]), (4, 5, 4))
        rows = read_jsonl(score_path("nsfw", "red"))
        self.assertEqual(len({r["image_id"] for r in rows}), 9)
        self.assertTrue((FILTERS_DIR / "nsfw" / "red.meta.json").exists())

    def test_cpu_guard(self):
        write_jsonl(score_path("nsfw", "guard"), [])
        det = RedDetector()
        det.name = "guard"
        from src.common import device
        old = device.CPU_LIMIT
        device.CPU_LIMIT = 3
        try:
            with self.assertRaises(SystemExit):
                run(det, device="cpu")
        finally:
            device.CPU_LIMIT = old

    def test_voting(self):
        scores = {"a": {"x": 0.9, "y": 0.2, "z": 0.6}, "b": {"x": 0.7, "y": 0.4, "z": 0.1}, "c": {"x": 0.1, "y": 0.8}}
        maj = combine(scores, "majority")
        self.assertEqual(set(maj), {"x", "y"})                 # only images scored by every model
        self.assertAlmostEqual(maj["x"], 2 / 3)
        self.assertAlmostEqual(combine(scores, "max")["y"], 0.8)
        self.assertAlmostEqual(combine(scores, "weighted", weights={"a": 2, "b": 1, "c": 1})["x"], (1.8 + 0.7 + 0.1) / 4)

    def test_metrics(self):
        scores = {"a": 0.9, "b": 0.8, "c": 0.3, "d": 0.1}
        labels = {"a": 1, "b": 0, "c": 1, "d": 0}
        self.assertEqual(prf(scores, labels, 0.5), {"n": 4, "precision": 0.5, "recall": 0.5, "f1": 0.5})
        self.assertEqual(best_f1(scores, labels)["threshold"], 0.3)


if __name__ == "__main__":
    unittest.main()
