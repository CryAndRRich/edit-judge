import itertools
import math
import random
import unittest

import tests.helpers  # noqa: F401  (temporary data folders)

from src.rank.finalize import best_order, estimate_weights
from src.scorer.data import split_of


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


if __name__ == "__main__":
    unittest.main()
