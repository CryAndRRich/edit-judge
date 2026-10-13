"""Per-edit features for the scorer (PROJECT.md Part II Step 3 and Step 6), computed for (original, edit, request)
and written to data/features/<feature>.jsonl keyed by image_id:
- preservation: CLIP image similarity, DINOv2 similarity, SSIM, LPIPS;
- change map: changed_area_ratio and edit_scope (local or global) from the aligned difference;
- request alignment: CLIP text-image similarity and CLIP direction similarity;
- quality: LAION aesthetic score; face identity similarity for portraits.
Every feature must also be computable for AI-generated edits, so no community metadata is used.
GPU work runs on Kaggle (see src/common/device.py).
"""
