import unittest

from tests.helpers import make_dataset  # sets the temporary data folders first

from src.instructions import prompts
from src.instructions.clients.base import Client
from src.instructions.parse import parse_answer, split_category
from src.instructions.run import output_path, run
from src.instructions.stats import agree, kappa, stats


class FakeClient(Client):
    """Answers like a VLM would, without a model."""
    name, model_id, batch_size = "fake", "test", 2

    def load(self, device):
        self.calls = 0

    def generate(self, prompt, image=None, max_tokens=256):
        self.calls += 1
        assert "{{INPUT}}" not in prompt and image is not None and image.exists()
        return 'Sure:\n```json\n{"instruction": "Remove the text.", "categories": ["Remove Text", "Enhance Media"]}\n```'


class ParseTests(unittest.TestCase):
    def test_categories_and_aliases(self):
        self.assertEqual(split_category("Change Color Background"), ("Change-Color", "Background"))
        self.assertEqual(split_category("Add Clothing"), ("Add", "Clothing/Accessories"))
        self.assertEqual(split_category("Open-Ended Other"), ("Open-Ended", "Other"))
        self.assertEqual(split_category("Formatting Media"), ("Image-Formatting", "Media"))
        self.assertEqual(split_category("Fly Person"), (None, None))

    def test_answer(self):
        r = parse_answer('{"instruction": "humorous.", "categories": ["Open-Ended Other", "Dance Person"]}',
                         "s", "m", "p")
        self.assertTrue(r.is_humorous)
        self.assertEqual(r.categories, [{"operation": "Open-Ended", "subject": "Other"}])
        self.assertEqual(r.problems, ["unknown category 'Dance Person'"])

    def test_fallback_without_json(self):
        r = parse_answer("Remove Person\nChange-Color Background", "s", "m", "p")
        self.assertEqual([c["operation"] for c in r.categories], ["Remove", "Change-Color"])
        self.assertIn("no JSON object", r.problems)


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        make_dataset(n_samples=6, n_edits=2)

    def test_prompt_file(self):
        text = prompts.load("analyze", 1)
        self.assertNotIn("# analyze_v1", text)
        self.assertIn("{{INPUT}}", text)
        self.assertIn("Change-Color: Color-correcting", text)

    def test_run_resume_and_stats(self):
        first = run(FakeClient(), sample=4, device="cpu")
        again = run(FakeClient(), sample=4, device="cpu")
        self.assertEqual((first["analysed"], again["analysed"], again["already_done"]), (4, 0, 4))
        self.assertTrue(output_path("fake", "analyze_v1").exists())
        s = stats("fake:analyze_v1")
        self.assertEqual(s["requests"], 4)
        self.assertEqual(s["operations_percent"]["Remove"]["ours"], 50.0)
        self.assertEqual(agree("fake:analyze_v1", "fake:analyze_v1")["same_categories"], 1.0)

    def test_kappa(self):
        self.assertAlmostEqual(kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]), 1.0)
        self.assertAlmostEqual(kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]), 0.0)


if __name__ == "__main__":
    unittest.main()
