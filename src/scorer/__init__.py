"""The learned scorer (PROJECT.md Part II Step 6): given an original, a request and an edit, a score such that
the edit people prefer scores higher. Trained on pairs (A better than B) of the same request; at inference every
edit gets its own score. Design and survey: docs/scorer-survey.md.

- data.py: preference pairs and final rankings (data/ranking/<version>/final/), and the split by sample (never by
  pair);
- backbones/: frozen encoders (CLIP ViT-L/14, PickScore, SigLIP, DINOv2); embed.py caches their embeddings;
- heads.py, train.py: scoring heads (linear or MLP) on embedding features, trained with a Bradley-Terry loss on
  hard or soft labels, or an HPSv3-style uncertainty loss (configs/scorer.toml);
- evaluate.py: pairwise accuracy (bootstrap interval), top-1, by source; zero-shot baselines (CLIP-score,
  similarity to the original, edit key order) and the human ceiling.
Outputs go to data/scorer/. Commands: python -m src.scorer --help.
"""
