import itertools
import json
import math
import unittest
from unittest import mock

import numpy as np
import torch

from tests.helpers import TMP, make_dataset  # sets the temporary data folders first

from src.paths import SCORER_DIR
from src.scorer import backbones
from src.scorer import embed as embed_module
from src.scorer import evaluate as evaluate_module
from src.scorer.backbones.base import Backbone
from src.scorer.data import load_pairs, load_rankings
from src.scorer.embed import embed, load_embeddings, save_embeddings, square_image
from src.scorer.evaluate import baselines, human_agreement, pair_metrics, paired_difference
from src.scorer.heads import pair_loss
from src.scorer.report import report
from src.scorer.rewards.base import RewardModel, edit_items, run as run_reward, scored_ids
from src.scorer.train import edit_features, train_run

DIM = 16
N_SAMPLES = 240
FAST = {"epochs": 60, "patience": 15, "lr": 1e-2, "seeds": [0, 1], "input_dropout": 0.0, "dropout": 0.1}


class ColourBackbone(Backbone):
    """Embeds an image by its mean colour and a text by its length: a stand-in for a real encoder."""
    name, model_id, image_size = "colour", "test", 16

    def load(self, device):
        self.device, self.dtype = device, torch.float32

    def encode_pixels(self, pixels):
        return pixels.mean(dim=(2, 3)) + 3.0

    def encode_texts(self, texts):
        return torch.tensor([[len(text), 1.0, 1.0] for text in texts])


class QualityReward(RewardModel):
    """Scores an edit by the synthetic quality of the tests, as a reward model would."""
    name, model_id, batch_size = "quality", "test", 3

    def __init__(self, quality):
        self.quality = quality

    def load(self, device):
        self.loaded = True

    def score(self, batch):
        assert all(original.mode == "RGB" and edit.mode == "RGB" for _, original, edit in batch)
        return [{"score": self.quality[item.image_id]} for item, _, _ in batch]


class ScorerTests(unittest.TestCase):
    """Synthetic data: random unit embeddings, an edit's hidden quality is its similarity to the text, and the
    final ranking and soft labels follow that quality."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        samples = make_dataset(n_samples=N_SAMPLES, n_edits=3)
        unit = lambda vector: vector / np.linalg.norm(vector)          # noqa: E731
        images, texts, cls.quality = {}, {}, {}
        final = TMP / "ranking" / "v1" / "final"
        final.mkdir(parents=True, exist_ok=True)
        with open(TMP / "pairs.jsonl", "w") as pairs_file, open(final / "rankings.jsonl", "w") as rankings_file:
            for number, sample in enumerate(samples):
                sample_id, keys = sample["id"], ["01", "02", "03"]
                texts[sample_id] = unit(rng.normal(size=DIM))
                images[f"{sample_id}/original"] = unit(rng.normal(size=DIM))
                for key in keys:
                    images[f"{sample_id}/{key}"] = unit(rng.normal(size=DIM))
                    cls.quality[f"{sample_id}/{key}"] = float(images[f"{sample_id}/{key}"] @ texts[sample_id])
                order = sorted(keys, key=lambda key: -cls.quality[f"{sample_id}/{key}"])
                noisy = [order[1], order[0], order[2]]
                rankings_file.write(json.dumps({
                    "id": sample_id, "source": "gier" if number % 4 == 0 else "realedit", "status": "ranked",
                    "ranking": order, "rankings": {"R1": order, "R2": order, "R3": noisy}}) + "\n")
                for better, worse in itertools.combinations(order, 2):    # the final dataset: a is preferred
                    gap = cls.quality[f"{sample_id}/{better}"] - cls.quality[f"{sample_id}/{worse}"]
                    pairs_file.write(json.dumps({"a": f"{sample_id}/{better}", "b": f"{sample_id}/{worse}",
                                                 "p": round(1 / (1 + math.exp(-8 * gap)), 4), "request": "test",
                                                 "instruction": "short"}) + "\n")
        (TMP / "index").mkdir(exist_ok=True)
        (TMP / "index" / "provenance.jsonl").write_text(json.dumps({"id": "s001", "trainable": False}) + "\n")
        ids = sorted(images)
        save_embeddings("synthetic", "images", ids, np.stack([images[image_id] for image_id in ids]))
        text_ids = sorted(texts)
        save_embeddings("synthetic", "texts-instruction", text_ids, np.stack([texts[key] for key in text_ids]))
        backbones.BACKBONES["synthetic"] = ".base:Backbone"       # known name, never loaded

    @classmethod
    def tearDownClass(cls):
        backbones.BACKBONES.pop("synthetic", None)

    def test_pairs_leave_out_untrainable_samples(self):
        ids = {pair["id"] for pair in load_pairs()}
        self.assertNotIn("s001", ids)
        self.assertEqual(len(ids), N_SAMPLES - 1)

    def test_metrics(self):
        pairs, rankings = load_pairs(split="test"), load_rankings()
        perfect = pair_metrics(pairs, self.quality, rankings)
        self.assertEqual((perfect["accuracy"], perfect["top1"]), (1.0, 1.0))
        reversed_scores = {key: -value for key, value in self.quality.items()}
        self.assertEqual(pair_metrics(pairs, reversed_scores, rankings)["accuracy"], 0.0)
        flat = pair_metrics(pairs, dict.fromkeys(self.quality, 0.0), rankings)
        self.assertEqual(flat["accuracy"], 0.5)
        self.assertAlmostEqual(flat["top1"], 1 / 3, places=3)
        self.assertEqual(set(perfect["by_source"]), {"gier", "realedit"})

    def test_paired_difference(self):
        pairs = load_pairs(split="test")
        flat = dict.fromkeys(self.quality, 0.0)
        better = paired_difference(pairs, self.quality, flat)
        self.assertAlmostEqual(better["difference"], 0.5, places=3)
        self.assertGreater(better["ci95"][0], 0.4)
        self.assertEqual(paired_difference(pairs, self.quality, self.quality)["difference"], 0.0)
        partial = dict(list(flat.items())[: len(flat) // 2])            # only the pairs both scored count
        self.assertLess(paired_difference(pairs, self.quality, partial)["pairs"], better["pairs"])

    def test_human_agreement(self):
        agreement = human_agreement(load_pairs(), load_rankings())
        # R3 swaps the two best edits: wrong on 1 of 3 pairs of every sample. Leave-one-out: R1 and R2 are
        # checked on the 2 pairs where the others agree (both right), R3 on all 3 (2 right)
        self.assertAlmostEqual(agreement["annotator_vs_final"], (3 + 3 + 2) / 9, places=3)
        self.assertAlmostEqual(agreement["annotator_vs_other_two"], 6 / 7, places=3)

    def test_uncertainty_loss(self):
        means = torch.tensor([2.0]), torch.tensor([0.0])
        sure = pair_loss(*means, torch.tensor([-4.0]), torch.tensor([-4.0]), torch.tensor([1.0]), "uncertainty")
        unsure = pair_loss(*means, torch.tensor([2.0]), torch.tensor([2.0]), torch.tensor([1.0]), "uncertainty")
        self.assertLess(float(sure), float(unsure))

    def test_features(self):
        features, ids = edit_features([("s000", "01"), ("s000", "02"), ("missing", "01")], ["synthetic"],
                                      ["vec", "sim"])
        self.assertEqual(ids, ["s000/01", "s000/02"])
        self.assertEqual(features.shape, (2, 3 * DIM + 6))

    def test_train_seeds_ensemble(self):
        for head, loss, inputs in (("linear", "soft", ["sim"]), ("mlp", "bt", ["vec", "sim"]),
                                   ("mlp", "uncertainty", ["vec", "sim"])):
            with self.subTest(head=head, loss=loss):
                result = train_run({"name": f"{head}-{loss}", "backbones": ["synthetic"], "inputs": inputs,
                                    "head": head, "loss": loss, **FAST}, device="cpu")
                self.assertGreater(result["ensemble_test"], 0.8)
                self.assertEqual(len(result["best_epochs"]), 2)
                self.assertIn("test_sd", result)
                saved = torch.load(SCORER_DIR / "runs" / f"{head}-{loss}" / "head.pt", weights_only=False)
                self.assertEqual([head_row["seed"] for head_row in saved["heads"]], [0, 1])

    def test_square_image_keeps_the_whole_image(self):
        from PIL import Image
        path = TMP / "wide.png"
        Image.new("RGB", (200, 50), (255, 0, 0)).save(path)
        square = square_image(path, size=64)
        self.assertEqual(square.shape, (64, 64, 3))
        self.assertEqual(tuple(square[0, 0]), (127, 127, 127))                  # grey padding above
        self.assertEqual(tuple(square[32, 0]), (255, 0, 0))                     # the image reaches both sides
        self.assertEqual(tuple(square[32, 63]), (255, 0, 0))

    def test_embed_shards_texts_and_cache(self):
        with mock.patch.object(embed_module, "get_backbone", lambda name: ColourBackbone()):
            first = embed(["colour"], limit=5, device="cpu", shard=(0, 2))
            second = embed(["colour"], limit=5, device="cpu", shard=(1, 2))
            again = embed(["colour"], limit=5, device="cpu")
        self.assertEqual(first["images"] + second["images"], 5 * 4)
        self.assertEqual(len(load_embeddings("colour", "images")), 20)          # both part files
        self.assertEqual(len(load_embeddings("colour", "texts-instruction")), 5)
        self.assertEqual(len(load_embeddings("colour", "texts-request")), 5)
        self.assertEqual(again["embedded"], [])                                 # cached: nothing to do
        self.assertEqual(again["cached"], ["colour"])

    def test_baselines_report_and_differences(self):
        train_run({"name": "report-run", "backbones": ["synthetic"], "inputs": ["vec", "sim"], **FAST},
                  device="cpu")
        with mock.patch.dict(evaluate_module.SIMILARITY_BASELINES, {"synthetic-score": "synthetic"}, clear=True):
            result = baselines()
        self.assertEqual(result["baselines"]["synthetic-score"]["test"]["accuracy"], 1.0)
        self.assertTrue((SCORER_DIR / "baselines" / "synthetic-score.jsonl").exists())
        summary = report()
        self.assertIsNotNone(summary["selected"])
        self.assertIn("synthetic-score", summary["selected_minus_baseline"])
        self.assertIn("| selected |", (SCORER_DIR / "report.md").read_text())

    def test_reward_runner_resumes_and_stops(self):
        items = edit_items("test")
        self.assertEqual(len({item.image_id for item in items}), len(items))
        self.assertEqual(items, edit_items("test"))                             # a fixed order
        self.assertTrue(all(item.text == "short" for item in items))
        model = QualityReward(self.quality)
        stopped = run_reward(model, device="cpu", allow_cpu=True, max_hours=0)    # time limit before any batch
        self.assertEqual((stopped["scored"], stopped["complete"]), (0, False))
        first = run_reward(model, device="cpu", limit=5)
        self.assertEqual((first["scored"], first["left"]), (5, len(items) - 5))
        shards = [run_reward(model, device="cpu", allow_cpu=True, shard=(index, 2)) for index in range(2)]
        self.assertEqual(sum(result["scored"] for result in shards), len(items) - 5)
        self.assertEqual(scored_ids("quality"), {item.image_id for item in items})
        self.assertTrue(run_reward(model, device="cpu")["complete"])          # nothing left
        self.assertEqual(evaluate_module.baseline_scores()["quality"], {item.image_id: self.quality[item.image_id]
                                                                        for item in items})


if __name__ == "__main__":
    unittest.main()
