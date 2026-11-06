import unittest
import zipfile

from tests.helpers import TMP, make_dataset  # sets the temporary data folders first

from src.process.pack import pack


class PackTests(unittest.TestCase):
    def test_pack_holds_only_the_final_dataset(self):
        make_dataset(n_samples=2, n_edits=2)
        (TMP / "index").mkdir(exist_ok=True)
        (TMP / "index" / "provenance.jsonl").write_text("{}\n")
        result = pack(TMP / "exports" / "test.zip")
        names = zipfile.ZipFile(result["zip"]).namelist()
        self.assertIn("pairs.jsonl", names)
        self.assertIn("images/s001/02.jpg", names)
        self.assertEqual(result["images"], len(list((TMP / "images").rglob("*.jpg"))))
        self.assertTrue(all(n == "pairs.jsonl" or n.startswith("images/") for n in names))


if __name__ == "__main__":
    unittest.main()
