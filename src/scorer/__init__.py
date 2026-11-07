"""The learned scorer (PROJECT.md Part II Step 6): given an original, a request and an edit, a score such that
the edit people prefer scores higher. Trained on pairs (A better than B) of the same request; at inference every
edit gets its own score. Design and results: ../docs/scorer.md.

- data.py: preference pairs of the final dataset (data/pairs.jsonl), rankings from them, and the split by sample
  (never by pair);
- backbones/: frozen encoders (CLIP ViT-L/14, PickScore, SigLIP, HPSv2.1, DINOv2); embed.py caches their
  embeddings, every image decoded once, on all GPUs;
- heads.py, train.py: scoring heads (linear or MLP) on embedding features, trained with a Bradley-Terry loss on
  hard or soft labels, or an HPSv3-style uncertainty loss, one head per seed (configs/scorer.toml);
- evaluate.py: pairwise accuracy (bootstrap interval), paired differences between scorers, top-1, the
  similarity baselines (CLIP-score, PickScore, SigLIP, HPSv2.1) and the human ceiling;
- rewards/: reward models and VLM judges scored on the test split (ImageReward, EditReward, Qwen2.5-VL judge,
  EditScore), each in its own environment, resumable across Kaggle sessions;
- report.py: one table of every run and baseline.
Outputs go to work/scorer/ (EJ_OUTPUT_DIR). Commands: python -m src.scorer --help.
"""
