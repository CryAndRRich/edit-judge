import unittest

from tests.helpers import make_dataset  # sets the temporary data folders first

from src.instructions import prompts
from src.instructions.clients.base import Client
from src.instructions.parse import loose_category, parse_answer, split_category
from src.instructions.run import output_path, result_files, run
from src.instructions.stats import agree, kappa, stats
from src.common.io import read_jsonl
from src.paths import INSTRUCTIONS_DIR


class FakeClient(Client):
    """Answers like a VLM would, without a model."""
    name, model_id, batch_size = "fake", "test", 2

    def load(self, device):
        self.calls = 0

    def generate(self, prompt, image=None, max_tokens=256):
        self.calls += 1
        assert "{{INPUT}}" not in prompt and image is not None and image.exists()
        return 'Sure:\n```json\n{"instruction": "Remove the text.", "categories": ["Remove Text", "Enhance Media"]}\n```'


class BrokenInBatchClient(FakeClient):
    """Gives "!!!!" for every request in a batch, a good answer when asked alone."""
    name = "broken"

    def generate_batch(self, prompts, images, max_tokens=256):
        return ["!" * 50 for _ in prompts]


class ParseTests(unittest.TestCase):
    def test_categories_and_aliases(self):
        self.assertEqual(split_category("Change Color Background"), ("Change-Color", "Background"))
        self.assertEqual(split_category("Add Clothing"), ("Add", "Clothing/Accessories"))
        self.assertEqual(split_category("Open-Ended Other"), ("Open-Ended", "Other"))
        self.assertEqual(split_category("Formatting Media"), ("Image-Formatting", "Media"))
        self.assertEqual(split_category("Remove People"), ("Remove", "Person"))
        self.assertEqual(split_category("Enhance Image"), ("Enhance", "Media"))
        self.assertEqual(split_category("Fly Person"), (None, None))

    def test_answer(self):
        r = parse_answer('{"instruction": "humorous.", "categories": ["Open-Ended Other", "Dance Person"]}',
                         "s", "m", "p")
        self.assertTrue(r.is_humorous)
        self.assertEqual(r.categories, [{"operation": "Open-Ended", "subject": "Other"},
                                        {"operation": "Other", "subject": "Person"}])
        self.assertEqual(r.problems, ["mapped 'Dance Person' to 'Other Person'"])

    def test_loose_labels(self):
        self.assertEqual(loose_category("Change-Color Helmet"), ("Change-Color", "Clothing/Accessories"))
        self.assertEqual(loose_category("Change-Clothing Person"), ("Replace", "Clothing/Accessories"))
        self.assertEqual(loose_category("Remove Reflection"), ("Enhance", "Other"))
        self.assertEqual(loose_category("Isolate Animal"), ("Remove", "Background"))
        self.assertEqual(loose_category("Enhance"), ("Enhance", "Other"))
        r = parse_answer('{"categories": ["Remove Person", "Other", "Change-Color"]}', "s", "m", "categorize_v1")
        self.assertEqual(r.categories, [{"operation": "Remove", "subject": "Person"},
                                        {"operation": "Change-Color", "subject": "Other"}])

    def test_split_pairs_and_extra_words(self):
        r = parse_answer('{"instruction": "x", "categories": ["Remove", "Background", "Add Person Background"]}',
                         "s", "m", "p")
        self.assertEqual(r.categories, [{"operation": "Remove", "subject": "Background"},
                                        {"operation": "Add", "subject": "Person"}])
        self.assertEqual(r.problems, ["extra words in 'Add Person Background'"])
        r = parse_answer('{"instruction": "x", "categories": ["Replace Person", "Person"]}', "s", "m", "p")
        self.assertEqual(len(r.categories), 1)

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
        self.assertIn("Never give an operation without a subject", prompts.load("analyze", 2))

    def test_run_resume_and_stats(self):
        first = run(FakeClient(), sample=4, device="cpu")
        again = run(FakeClient(), sample=4, device="cpu")
        self.assertEqual((first["analysed"], again["analysed"], again["already_done"]), (4, 0, 4))
        self.assertTrue(output_path("fake", "analyze_v1").exists())
        s = stats("fake:analyze_v1")
        self.assertEqual(s["requests"], 4)
        self.assertEqual(s["operations_percent"]["Remove"]["ours"], 50.0)
        self.assertEqual(agree("fake:analyze_v1", "fake:analyze_v1")["same_categories"], 1.0)

    def test_shards_cover_all_once(self):
        a = run(FakeClient(), version=2, device="cpu", shard=(0, 2))
        b = run(FakeClient(), version=2, device="cpu", shard=(1, 2))
        self.assertEqual(a["analysed"] + b["analysed"], 6)
        self.assertEqual(len(result_files("fake", "analyze_v2")), 2)
        self.assertEqual(stats("fake:analyze_v2")["requests"], 6)
        self.assertEqual(run(FakeClient(), version=2, device="cpu")["analysed"], 0)   # all done, any shard file

    def test_broken_batch_answers_asked_again(self):
        r = run(BrokenInBatchClient(), sample=4, device="cpu")
        self.assertEqual((r["retried_alone"], r["with_problems"]), (4, 0))

    def test_time_limit(self):
        client = FakeClient()
        client.name = "timed"
        r = run(client, sample=4, device="cpu", max_hours=0)
        self.assertEqual((r["analysed"], r["left"]), (0, 4))

    def test_kappa(self):
        self.assertAlmostEqual(kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]), 1.0)
        self.assertAlmostEqual(kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]), 0.0)

    def test_bare_subject_takes_the_previous_operation(self):
        row = parse_answer('{"instruction": "Remove the person.", "categories": ["Remove Background", "Person"]}',
                           "s1", "m", "analyze_v2")
        self.assertEqual(row.categories, [{"operation": "Remove", "subject": "Background"},
                                          {"operation": "Remove", "subject": "Person"}])
        row = parse_answer('{"categories": ["Rotate Background"]}', "s1", "m", "categorize_v1")
        self.assertEqual(row.problems, ["mapped 'Rotate Background' to 'Image-Formatting Background'"])

    def test_complete_uses_realedit_instructions_and_resumes(self):
        """Samples with a RealEdit instruction get categories only; the others the full prompt; a second call
        goes on where the first stopped; the merge takes RealEdit's instruction where there is one."""
        from unittest import mock
        from src.instructions import complete as completion
        make_dataset(n_samples=6, n_edits=2)
        realedit = {"s000": "Restore this photo.", "s001": "humorous", "s002": "Remove the dog."}
        client = FakeClient()
        client.name = "completion"
        prompts_seen = []
        original = client.generate

        def generate(prompt, image=None, max_tokens=256):
            prompts_seen.append("instruction" in prompt.split("Answer with")[1])
            return original(prompt, image, max_tokens)

        client.generate = generate
        with mock.patch.object(completion, "realedit_instructions", lambda: realedit):
            first = completion.complete(client, device="cpu", limit=2)
            self.assertEqual(first["merged"]["done"], 4)
            self.assertEqual(first["merged"]["left"], {"analyze_v2": 1, "categorize_v1": 1})
            second = completion.complete(client, device="cpu")
            self.assertTrue(second["merged"]["complete"])
        self.assertEqual(sorted(prompts_seen), [False] * 3 + [True] * 3)      # 3 full prompts, 3 categories only
        final = {row["id"]: row for row in read_jsonl(INSTRUCTIONS_DIR / "completion" / "final.jsonl")}
        self.assertEqual(final["s001"]["instruction"], "humorous")
        self.assertEqual(final["s001"]["instruction_source"], "realedit")
        self.assertEqual(final["s004"]["instruction"], "Remove the text.")
        self.assertEqual(final["s004"]["instruction_source"], "qwen")
        self.assertEqual(final["s004"]["categories"], ["Remove Text/Patterns", "Enhance Media"])


if __name__ == "__main__":
    unittest.main()
