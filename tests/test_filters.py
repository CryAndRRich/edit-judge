import subprocess
import sys
import unittest

from PIL import Image, ImageStat

from tests.helpers import make_dataset  # sets the temporary data folders first

from src.common.dataset import iter_images
from src.common.io import read_jsonl, write_jsonl
from src.filters import TASKS, get_detector, registry
from src.filters.base import Detector, run, score_files, score_path
from src.filters.benchmark import bench_dir, labels_path
from src.filters.evaluate import average_precision, best_f1, evaluate, precision_recall_f1, roc_auc, write_markdown
from src.filters.voting import combine, load_scores
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
        self.assertEqual(precision_recall_f1(scores, labels, 0.5), {"n": 4, "precision": 0.5, "recall": 0.5, "f1": 0.5})
        self.assertEqual(best_f1(scores, labels)["threshold"], 0.3)

    def test_ranking_metrics(self):
        labels = {"a": 1, "b": 1, "c": 0, "d": 0}
        self.assertEqual(roc_auc({"a": 0.9, "b": 0.4, "c": 0.4, "d": 0.1}, labels), 0.875)   # one tie counts 1/2
        self.assertEqual(average_precision({"a": 0.9, "b": 0.3, "c": 0.5, "d": 0.1}, labels), round((1 + 2 / 3) / 2, 4))

    def test_registered_detectors(self):
        """Every registered detector loads its class without importing any model library."""
        for task in TASKS:
            self.assertTrue(registry(task))
            for name in registry(task):
                det = get_detector(task, name)
                self.assertEqual((det.task, det.name), (task, name))
                self.assertIsInstance(det.install, tuple)
        from src.filters.envs import env_name
        nemo = get_detector("nsfw", "nemo")
        self.assertTrue(nemo.isolated)                       # its own torch: not on top of the machine's packages
        self.assertNotEqual(env_name(nemo.install, True), env_name(nemo.install, False))
        check = ("import sys; from src.filters import TASKS, get_detector, registry; "
                 "[get_detector(t, n) for t in TASKS for n in registry(t)]; "
                 "print(sorted(m for m in ('torch', 'tensorflow', 'timm', 'ultralytics', 'nudenet', 'nemo_curator', "
                 "'open_clip', 'easyocr') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, check=True).stdout
        self.assertEqual(out.strip(), "[]")                 # in a fresh process: other tests import torch

    def test_benchmark_run_and_evaluation(self):
        folder = bench_dir("watermark")
        rows = []
        for cls, red, label in (("watermark", 250, 1), ("watermark", 200, 1), ("clean", 120, 0), ("clean", 10, 0)):
            (folder / "pub" / cls).mkdir(parents=True, exist_ok=True)
            file = f"pub/{cls}/{red}.jpg"
            Image.new("RGB", (32, 32), (red, 0, 0)).save(folder / file)
            rows.append({"image_id": f"pub/{cls}/{red}", "label": label, "file": file})
        write_jsonl(labels_path("watermark"), rows)
        det = RedDetector()
        det.task = "watermark"
        result = run(det, device="cpu", on="benchmark", latency=3)
        self.assertEqual(result["scored"], 4)
        self.assertEqual(result["latency_ms"]["n"], 3)
        report = evaluate("watermark", {}, on="benchmark")
        red = report["models"]["red"]
        self.assertEqual((red["tp"], red["fp"], red["roc_auc"]), (2, 0, 1.0))
        self.assertEqual(red["flagged_per_class"], {"pub/clean": 0.0, "pub/watermark": 1.0})
        self.assertEqual(red["by_set"]["pub"]["f1"], 1.0)
        for name in ("half", "noise"):                        # two weaker models: the vote of all three is worse
            det = RedDetector()
            det.task, det.name = "watermark", name
            det.score = (lambda images: [0.6] * len(images)) if name == "half" else \
                (lambda images: [0.9 if i % 2 else 0.1 for i, _ in enumerate(images)])
            run(det, device="cpu", on="benchmark")
        report = evaluate("watermark", {}, on="benchmark")
        self.assertEqual(report["choice"]["single"]["model"], "red")
        self.assertEqual(report["choice"]["f1_gain"], 0.0)
        self.assertIn("red", report["subset_votes"][0]["models"])
        self.assertEqual(len(report["subset_votes"]), 12)      # 4 subsets x 3 methods (no weights: no "weighted")
        text = write_markdown(report)
        self.assertIn("| red | 1.000 |", text)
        self.assertTrue((FILTERS_DIR / "watermark" / "benchmark" / "evaluation.md").exists())

    def test_shards_cover_every_image_once(self):
        for i in range(3):
            det = RedDetector()
            det.name = "sharded"
            run(det, device="cpu", shard=(i, 3))
        ids = [r["image_id"] for f in score_files("nsfw", "sharded") for r in read_jsonl(f)]
        self.assertEqual(sorted(ids), sorted(r.image_id for r in iter_images()))
        self.assertEqual(len(load_scores("nsfw", ["sharded"])["sharded"]), 9)

    def test_multirun_schedules_over_gpus(self):
        """Fake processes: on the dataset every model is split into one shard per GPU, on the benchmark the models
        run side by side; a second shard starts only after the first one has loaded the model."""
        from unittest import mock
        from src.filters import parallel
        calls = []

        class FakeProc:
            def __init__(self, cmd, cwd=None, env=None, stdout=None, stderr=None, text=None):
                calls.append((cmd, env["CUDA_VISIBLE_DEVICES"]))
                stdout.write("loaded\n")
                stdout.flush()
                self.returncode = 0

            def poll(self):
                return 0

        with mock.patch.object(parallel.subprocess, "Popen", FakeProc), \
                mock.patch.object(parallel, "ensure_env", lambda install, log=None, isolated=False: sys.executable):
            out = parallel.multirun("watermark", "dataset", ["convnext", "laion"], gpus=2)
            self.assertEqual(sorted(out["jobs"]), ["watermark-convnext-0", "watermark-convnext-1",
                                                   "watermark-laion-0", "watermark-laion-1"])
            shards = sorted(c[0][c[0].index("--shard") + 1] for c in calls)
            self.assertEqual(shards, ["0/2", "0/2", "1/2", "1/2"])
            calls.clear()
            out = parallel.multirun("watermark", "benchmark", ["convnext", "laion", "ocr"], gpus=2, latency=20)
            self.assertEqual(out["failed"], {})
            self.assertEqual({c[1] for c in calls}, {"0", "1"})
            self.assertTrue(all("--latency" in c[0] and "--shard" not in c[0] for c in calls))
            first = {c[0][c[0].index("--model") + 1] for c in calls[:2]}
            self.assertEqual(first, {"ocr", "laion"})                                   # slow models first

    def test_benchmark_sampling_and_zip_set(self):
        from src.filters import benchmark
        idx = benchmark.sample_indices(["a"] * 5 + ["b"] * 2, per_class=3)
        self.assertEqual(len(idx), 5)
        self.assertEqual(idx, benchmark.sample_indices(["a"] * 5 + ["b"] * 2, per_class=3))   # fixed seed
        self.assertEqual(idx[-2:], [5, 6])                                              # small class kept whole
        import io
        import zipfile
        from unittest import mock
        archive = bench_dir("nsfw") / "_fake.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as zf:
            for cls in ("nsfw", "sfw"):
                for k in range(3):
                    buf = io.BytesIO()
                    Image.new("RGB", (8, 8)).save(buf, "JPEG")
                    zf.writestr(f"nsfw_detection_xl/{cls}/{k}.jpg", buf.getvalue())
            zf.writestr("nsfw_detection_xl/.git/config", "x")
        with mock.patch("huggingface_hub.hf_hub_download", lambda *a, **k: str(archive)), \
                mock.patch.object(benchmark, "PER_CLASS", 2):
            rows = benchmark.nsfw_detection_xl(archive.parent)
        self.assertEqual(sorted((r.cls, r.label) for r in rows), [("nsfw", 1)] * 2 + [("sfw", 0)] * 2)


if __name__ == "__main__":
    unittest.main()
