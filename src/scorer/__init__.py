"""The learned scorer (PROJECT.md Part II Step 6): given an original, a request and an edit, a score such that
the edit people prefer scores higher.

- data.py: preference pairs from the final team rankings (data/ranking/<version>/final/pairs.jsonl), joined with
  the features, and the split by sample (never by pair);
- models: Bradley-Terry on feature differences (logistic regression, gradient-boosted trees, small MLP);
- evaluation: pairwise accuracy on held-out samples against random, SSIM-only, aesthetic-only and a VLM judge.
Outputs go to data/scorer/.
"""
