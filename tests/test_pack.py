import json
import unittest
import zipfile

from tests.helpers import TMP, make_dataset  # sets the temporary data folders first

from src.process.pack import pack


class PackTests(unittest.TestCase):
    def test_pack_leaves_out_private_files(self):
        make_dataset(n_samples=2, n_edits=2)
        (TMP / "index").mkdir(exist_ok=True)
        (TMP / "index" / "provenance.jsonl").write_text("{}\n")
        (TMP / "index" / "ledger.jsonl").write_text("{}\n")
        v1 = TMP / "ranking" / "v1"
        for sub in ("final", "labels", "results/R1"):
            (v1 / sub).mkdir(parents=True, exist_ok=True)
        (v1 / "manifest.json").write_text(json.dumps({"chunks": {}}))
        (v1 / "final" / "pairs.jsonl").write_text("{}\n")
        (v1 / "labels" / "summary.json").write_text(json.dumps({"names": ["someone"]}))
        (v1 / "results" / "R1" / "v1_c01__someone.json").write_text("{}")
        (TMP / "raw").mkdir(exist_ok=True)
        (TMP / "raw" / ".salt").write_text("secret")
        result = pack(TMP / "exports" / "test.zip")
        names = zipfile.ZipFile(result["zip"]).namelist()
        self.assertIn("samples/s001/edits/02.jpg", names)
        self.assertIn("ranking/v1/final/pairs.jsonl", names)
        self.assertEqual(result["images"], len(list((TMP / "samples").rglob("*.jpg"))))
        for private in ("raw/.salt", "index/ledger.jsonl", "ranking/v1/labels/summary.json",
                        "ranking/v1/results/R1/v1_c01__someone.json"):
            self.assertNotIn(private, names)


if __name__ == "__main__":
    unittest.main()
