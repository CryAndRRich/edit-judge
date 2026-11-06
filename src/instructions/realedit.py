"""RealEdit's published simplified instructions (CVPR 2025, CC BY 4.0): the `instruction` column of its two CSV
files on Hugging Face (peter-sushko/RealEdit, pinned commit), written by the authors' LLM with the rewriting prompt
of Fig. 19. RealEdit does not publish the categories.

Our sample ID of a RealEdit post is sha1("realedit:<post id>")[:12] (src/process/build.py), so the instructions
can be joined to the final dataset without the provenance index, e.g. on Kaggle.
"""

import csv
import hashlib

REPO, REVISION = "peter-sushko/RealEdit", "e3a1da4e7a316a3aa4e302a582df3fee43d30db8"
FILES = ("RealEdit_train_split_urls.csv", "RealEdit_test_split_urls.csv")


def realedit_sample_id(post_id: str) -> str:
    return hashlib.sha1(f"realedit:{post_id}".encode("utf-8")).hexdigest()[:12]


def realedit_instructions() -> dict[str, str]:
    """Sample ID -> RealEdit's instruction ("humorous" included), for every post with a non-empty instruction."""
    from huggingface_hub import hf_hub_download
    instructions = {}
    for name in FILES:
        path = hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION)
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                text = (row["instruction"] or "").strip()
                if text:
                    instructions.setdefault(realedit_sample_id(row["input_image_name"].rsplit(".", 1)[0]), text)
    return instructions
