import unittest

import tests.helpers  # noqa: F401

from src.instructions.clients.base import parse_json
from src.instructions.schema import RequestAnalysis


class InstructionTests(unittest.TestCase):
    def test_schema_checks_labels(self):
        a = RequestAnalysis("s1", "m", "analyze_v1", True, "Remove the person", operations=["Delete", "Erase"],
                            subjects=["Person"], ambiguity=7)
        self.assertEqual(a.problems(), ["operation 'Erase'", "ambiguity 7"])

    def test_parse_json(self):
        self.assertEqual(parse_json('Sure! ```json\n{"a": [1, 2]}\n```'), {"a": [1, 2]})


if __name__ == "__main__":
    unittest.main()
