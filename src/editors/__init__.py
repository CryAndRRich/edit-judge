"""Editing models and their benchmark (PROJECT.md Part II Step 7): one wrapper per model with the same interface
(original image, instruction, optional mask -> edited image), grouped by the operations they serve (Delete:
erasers; Add/Replace/Transform/Apply: inpainting; Restore/Superresolution: restoration; Recoloring; Relighting;
plus generalist models). The benchmark scores every model's output with the scorer and writes the ranking table
to data/editors/. All runs are on Kaggle or through APIs.
"""
