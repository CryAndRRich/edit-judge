import itertools
import json
import math
import unittest

import numpy as np
import torch

from tests.helpers import TMP, make_dataset  # sets the temporary data folders first

from src.paths import SCORER_DIR
from src.scorer import backbones
from src.scorer.backbones.base import Backbone
from src.scorer.data import load_pairs, load_rankings
from src.scorer.embed import embed, load_embeddings, save_embeddings
from src.scorer.evaluate import baselines, human_agreement, pair_metrics
from src.scorer.heads import pair_loss
from src.scorer.train import edit_features, report, train_run

DIM = 16
N_SAMPLES = 240


class ColourBackbone(Backbone):
    """Embeds an image by its mean colour and a text by its length: a stand-in for a real encoder."""
    name, model_id, batch_size = "colour", "test", 8

    def load(self, device):
        self.device = device

    def preprocess(self, image):
        return torch.tensor(np.asarray(image, dtype=np.float32).mean(axis=(0, 1)) + 1)

    def encode_images(self, pixels):
        return torch.nn.functional.normalize(pixels.float(), dim=-1).numpy()

    def encode_texts(self, texts):
        return np.array([[len(t), 1.0, 1.0] for t in texts]) / np.sqrt([[len(t) ** 2 + 2] for t in texts])


class ScorerTests(unittest.TestCase):
    """Synthetic data: random unit embeddings, an edit's hidden quality is its similarity to the request, and
    the final ranking and soft labels follow that quality."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        samples = make_dataset(n_samples=N_SAMPLES, n_edits=3)
        unit = lambda v: v / np.linalg.norm(v)
        images, texts, cls.quality = {}, {}, {}
        final = TMP / "ranking" / "v1" / "final"
        final.mkdir(parents=True, exist_ok=True)
        with open(final / "pairs.jsonl", "w") as fp, open(final / "rankings.jsonl", "w") as fr:
            for n, s in enumerate(samples):
                sid, keys = s["id"], ["01", "02", "03"]
                texts[sid] = unit(rng.normal(size=DIM))
                images[f"{sid}/original"] = unit(rng.normal(size=DIM))
                for k in keys:
                    images[f"{sid}/{k}"] = unit(rng.normal(size=DIM))
                    cls.quality[f"{sid}/{k}"] = float(images[f"{sid}/{k}"] @ texts[sid])
                order = sorted(keys, key=lambda k: -cls.quality[f"{sid}/{k}"])
                noisy = [order[1], order[0], order[2]]
                fr.write(json.dumps({"id": sid, "source": "gier" if n % 4 == 0 else "realedit", "status": "ranked",
                                     "ranking": order, "rankings": {"R1": order, "R2": order, "R3": noisy}}) + "\n")
                for a, b in itertools.combinations(keys, 2):
                    p_a = 1 / (1 + math.exp(-8 * (cls.quality[f"{sid}/{a}"] - cls.quality[f"{sid}/{b}"])))
                    fp.write(json.dumps({"id": sid, "a": a, "b": b, "p_a": round(p_a, 4), "status": "ranked",
                                         "winner": a if order.index(a) < order.index(b) else b}) + "\n")
        (TMP / "index").mkdir(exist_ok=True)
        (TMP / "index" / "provenance.jsonl").write_text(json.dumps({"id": "s001", "trainable": False}) + "\n")
        ids = sorted(images)
        save_embeddings("synthetic", "images", ids, np.stack([images[i] for i in ids]))
        save_embeddings("synthetic", "texts", sorted(texts), np.stack([texts[i] for i in sorted(texts)]))
        backbones.BACKBONES["synthetic"] = ".base:Backbone"       # known name, never loaded

    @classmethod
    def tearDownClass(cls):
        backbones.BACKBONES.pop("synthetic", None)

    def test_pairs_leave_out_untrainable_samples(self):
        ids = {p["id"] for p in load_pairs("v1")}
        self.assertNotIn("s001", ids)
        self.assertEqual(len(ids), N_SAMPLES - 1)

    def test_metrics(self):
        pairs, rankings = load_pairs("v1", "test"), load_rankings("v1")
        perfect = pair_metrics(pairs, self.quality, rankings)
        self.assertEqual(perfect["accuracy"], 1.0)
        self.assertEqual(perfect["top1"], 1.0)
        self.assertLessEqual(perfect["ci95"][0], 1.0)
        self.assertEqual(pair_metrics(pairs, {k: -v for k, v in self.quality.items()}, rankings)["accuracy"], 0.0)
        flat = pair_metrics(pairs, dict.fromkeys(self.quality, 0.0), rankings)
        self.assertEqual(flat["accuracy"], 0.5)
        self.assertAlmostEqual(flat["top1"], 1 / 3, places=3)
        self.assertEqual(set(perfect["by_source"]), {"gier", "realedit"})

    def test_human_agreement(self):
        h = human_agreement(load_pairs("v1"), load_rankings("v1"))
        # R3 swaps the two best edits: wrong on 1 of 3 pairs of every sample. Leave-one-out: R1 and R2 are
        # checked on the 2 pairs where the others agree (both right), R3 on all 3 (2 right)
        self.assertAlmostEqual(h["annotator_vs_final"], (3 + 3 + 2) / 9, places=3)
        self.assertAlmostEqual(h["annotator_vs_other_two"], 6 / 7, places=3)

    def test_uncertainty_loss(self):
        mu = torch.tensor([2.0]), torch.tensor([0.0])
        sure = pair_loss(*mu, torch.tensor([-4.0]), torch.tensor([-4.0]), torch.tensor([1.0]), "uncertainty")
        unsure = pair_loss(*mu, torch.tensor([2.0]), torch.tensor([2.0]), torch.tensor([1.0]), "uncertainty")
        self.assertLess(float(sure), float(unsure))

    def test_features(self):
        x, ids = edit_features([("s000", "01"), ("s000", "02"), ("missing", "01")], ["synthetic"], ["vec", "sim"])
        self.assertEqual(ids, ["s000/01", "s000/02"])
        self.assertEqual(x.shape, (2, 3 * DIM + 6))

    def test_train_heads_learn_the_hidden_quality(self):
        for head, loss, inputs in (("linear", "soft", ["sim"]), ("mlp", "bt", ["vec", "sim"]),
                                   ("mlp", "uncertainty", ["vec", "sim"])):
            with self.subTest(head=head, loss=loss):
                result = train_run({"name": f"{head}-{loss}", "backbones": ["synthetic"], "inputs": inputs,
                                    "head": head, "loss": loss, "epochs": 60, "patience": 15}, device="cpu")
                self.assertGreater(result["test"]["accuracy"], 0.8)
                self.assertTrue((SCORER_DIR / "runs" / f"{head}-{loss}" / "head.pt").exists())

    def test_embed_baselines_and_report(self):
        result = embed(ColourBackbone(), limit=5, device="cpu")
        self.assertEqual(result["images"], 5 * 4)
        self.assertEqual(len(load_embeddings("colour", "images")), 20)
        self.assertEqual(len(load_embeddings("colour", "texts")), 5)
        b = baselines("v1")
        self.assertIn("synthetic:edit-request", b["baselines"])
        self.assertEqual(b["baselines"]["synthetic:edit-request"]["test"]["accuracy"], 1.0)
        self.assertIn("edit-key-order", b["baselines"])
        rows = report()["rows"]
        self.assertTrue(any(r["name"] == "synthetic:edit-request" for r in rows))
        self.assertTrue((SCORER_DIR / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
