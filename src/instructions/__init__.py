"""Request analysis with open VLMs, following RealEdit directly (PROJECT.md Part II Steps 1a, 2 and 4): for each
request, one call returns the simplified instruction (or "humorous") and RealEdit's "operation subject"
categories, one per task in the request.

- schema.py: the result and RealEdit's label sets; parse.py: reading the model's answer;
- prompts/: versioned prompt files (analyze_v1 = RealEdit's two prompts in one call);
- clients/: open VLMs (Qwen2.5-VL 3B and 7B) with one interface;
- run.py: the resumable runner; stats.py: distribution vs RealEdit and agreement between runs.
Results: data/instructions/<client>/<prompt>.jsonl. The raw request is always kept; simplified instructions are
never used as a training signal.
"""
