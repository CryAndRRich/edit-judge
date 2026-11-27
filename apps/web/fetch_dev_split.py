"""Downloads the 5 sample photos of the web app (apps/web/studio/) into apps/web/dev_split/.

No image is generated: every photo is a real photograph from a public source.
- Lorem Picsum (https://picsum.photos): photos from Unsplash, under the Unsplash License; the author and the
  Unsplash page come from the Picsum info endpoint.
- COCO val2017 (https://cocodataset.org): Flickr photos under the Creative Commons licenses listed by COCO.

Each sample also gets a request, one per task (remove, add_replace, change_color, enhance, formatting), so the
pipeline can be run end to end; the GPU server answers them ahead of time (run.py --precompute) and keeps the answers
in its result cache, so on the demo the sample images show their edits at once. dev_split/manifest.json records the
source, author, license, task and request of every photo. The photos are committed (public, no people); this
script is needed only to download them again (--force) or after changing SAMPLES.

Usage: python apps/web/fetch_dev_split.py [--force]
"""

import argparse
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

OUT = Path(__file__).resolve().parent / "dev_split"
HEADERS = {"User-Agent": "edit-judge-dev-split/1.0"}
MAX_SIDE = 1024

SAMPLES = [                 # one photo per task (5 of the 7; no people), the sample images of the page in this order
    {"id": "coco-000000039769", "kind": "coco", "coco_file": "000000039769.jpg", "task": "remove",
     "instruction": "Xóa cái điều khiển TV khỏi ảnh"},
    {"id": "picsum-1084", "kind": "picsum", "picsum_id": 1084, "task": "add_replace",
     "instruction": "Add a red Santa hat on the walrus on the right"},
    {"id": "picsum-1015", "kind": "picsum", "picsum_id": 1015, "task": "change_color",
     "instruction": "Đổi màu bầu trời thành màu cam như hoàng hôn"},
    {"id": "picsum-1074", "kind": "picsum", "picsum_id": 1074, "task": "enhance",
     "instruction": "Làm ảnh sáng hơn và rõ nét hơn"},
    {"id": "picsum-1025", "kind": "picsum", "picsum_id": 1025, "task": "formatting",
     "instruction": "Remove the background"},
]


def get(url: str, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} ({error})", flush=True)
            time.sleep(2 * (attempt + 1))
    raise AssertionError


def fetch(sample: dict) -> dict:
    if sample["kind"] == "picsum":
        info = get(f"https://picsum.photos/id/{sample['picsum_id']}/info").json()
        width, height = info["width"], info["height"]
        scale = min(1.0, MAX_SIDE / max(width, height))
        url = f"https://picsum.photos/id/{sample['picsum_id']}/{round(width * scale)}/{round(height * scale)}.jpg"
        meta = {"source": "Lorem Picsum (Unsplash)", "author": info.get("author"), "page": info.get("url"),
                "license": "Unsplash License (https://unsplash.com/license)"}
    else:
        url = f"http://images.cocodataset.org/val2017/{sample['coco_file']}"
        meta = {"source": "COCO val2017", "author": None, "page": "https://cocodataset.org",
                "license": "Creative Commons (per-image license in the COCO annotations)"}
    image = Image.open(BytesIO(get(url).content))
    image.load()
    return {"image": image.convert("RGB"), "url": url, **meta}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download again even if the photo exists")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    manifest = []
    for sample in SAMPLES:
        path = OUT / f"{sample['id']}.jpg"
        entry = {"id": sample["id"], "file": path.name, "task": sample["task"], "instruction": sample["instruction"]}
        old = {}
        manifest_path = OUT / "manifest.json"
        if manifest_path.exists():
            old = {row["id"]: row for row in json.loads(manifest_path.read_text())}
        if path.exists() and not args.force and sample["id"] in old:
            print(f"keep {path.name}")
            manifest.append({**old[sample["id"]], "task": sample["task"], "instruction": sample["instruction"]})
            continue
        print(f"download {sample['id']} ...", flush=True)
        try:
            result = fetch(sample)
        except Exception as error:          # one failed source must not stop the others
            print(f"  FAILED: {error}", file=sys.stderr)
            continue
        image = result.pop("image")
        image.save(path, quality=92)
        entry.update(result, width=image.width, height=image.height)
        manifest.append(entry)
        print(f"  saved {path.name} ({image.width}x{image.height})")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(manifest)}/{len(SAMPLES)} photos in {OUT}")
    return 0 if len(manifest) >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
