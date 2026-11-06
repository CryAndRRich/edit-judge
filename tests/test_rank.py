import itertools
import math
import random
import unittest

import json

from tests.helpers import TMP, make_dataset  # sets the temporary data folders first

from src.common.dataset import iter_samples
from src.paths import FINAL_PAIRS
from src.rank.export import export
from src.rank.finalize import best_order, estimate_weights
from src.scorer.data import load_pairs, split_of


class RankTests(unittest.TestCase):
    def test_best_order_is_optimal(self):
        rng = random.Random(1)
        for _ in range(100):
            keys = [f"{i:02d}" for i in range(1, rng.randint(2, 6) + 1)]
            logp = {}
            for x, y in itertools.combinations(keys, 2):
                q = rng.random()
                logp[(x, y)], logp[(y, x)] = math.log(q), math.log(1 - q)
            score = lambda o: sum(logp[(a, b)] for i, a in enumerate(o) for b in o[i + 1:])
            self.assertAlmostEqual(score(best_order(keys, logp, keys)),
                                   max(score(o) for o in itertools.permutations(keys)))

    def test_reliability(self):
        """A voter that always agrees with the truth gets more weight than a random one."""
        rng = random.Random(2)
        units = []
        for _ in range(3000):
            truth = rng.choice((1, -1))
            units.append({"good": truth, "ok": truth if rng.random() < 0.75 else -truth,
                          "random": rng.choice((1, -1))})
        w, p = estimate_weights(units)
        self.assertGreater(w["good"], w["ok"])
        self.assertLess(w["random"], 0.1)

    def test_split_is_stable(self):
        counts = {"train": 0, "dev": 0, "test": 0}
        for i in range(10000):
            counts[split_of(f"{i:012x}")] += 1
        self.assertEqual(split_of("abc"), split_of("abc"))
        self.assertTrue(6700 < counts["train"] < 7300 and 1300 < counts["dev"] < 1700)

    def test_export_keeps_only_ranked_samples(self):
        make_dataset(n_samples=3, n_edits=3)
        final = TMP / "ranking" / "v1" / "final"
        final.mkdir(parents=True, exist_ok=True)
        rows = [{"id": "s000", "status": "ranked", "ranking": ["02", "01", "03"]},
                {"id": "s001", "status": "skipped", "ranking": None},
                {"id": "s002", "status": "ranked", "ranking": ["03", "02", "01"]}]

        def write_final():
            (final / "rankings.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            pairs = [{"id": r["id"], "a": a, "b": b,
                      "p_a": 0.7 if r["ranking"].index(a) < r["ranking"].index(b) else 0.3}
                     for r in rows if r["ranking"] for a, b in itertools.combinations(["01", "02", "03"], 2)]
            pairs[0]["p_a"] = 0.45 if pairs[0]["p_a"] == 0.7 else 0.55    # one pair against the ranking
            (final / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
        write_final()
        result = export()
        self.assertEqual((result["samples"], result["pairs"]), (2, 6))
        self.assertEqual(sorted(p.name for p in (TMP / "images").iterdir()), ["s000", "s002"])
        self.assertTrue((TMP / "unranked" / "s001" / "edits" / "03.jpg").exists())
        lines = [json.loads(l) for l in FINAL_PAIRS.read_text().splitlines()]
        self.assertEqual([(l["a"], l["b"], l["p"]) for l in lines[:3]],
                         [("s000/01", "s000/02", 0.55), ("s000/02", "s000/03", 0.7), ("s000/01", "s000/03", 0.7)])
        self.assertTrue(all(l["p"] >= 0.5 for l in lines))
        sample = next(iter_samples())
        self.assertEqual([e["image"] for e in sample["edits"]], ["01.jpg", "02.jpg", "03.jpg"])
        self.assertEqual(load_pairs(split=None)[0]["winner"], "01")
        rows[1] = {"id": "s001", "status": "ranked", "ranking": ["01", "02", "03"]}   # ranked by a later finalize
        write_final()
        self.assertEqual(export()["restore_from_unranked"], 1)
        self.assertTrue((TMP / "images" / "s001" / "03.jpg").exists())


if __name__ == "__main__":
    unittest.main()
