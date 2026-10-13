"""LLM/VLM analysis of the edit requests (PROJECT.md Part II Steps 1a, 2 and 4): is it an edit request, a short
clean instruction, its parts, the operations and subject, ambiguity, language. The output of every model goes to
data/instructions/<model>/analyses.jsonl, one RequestAnalysis per sample (schema.py).

- prompts/: prompt templates with a version number (<name>_v<k>.txt), loaded by prompts.load();
- clients/: one class per model or API, all with the same `generate` method (clients/base.py).
The raw request text is always kept: cleaned instructions are never a training signal.
"""
