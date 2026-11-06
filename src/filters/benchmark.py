"""Public benchmark sets with known labels, to compare detectors before they run on the dataset.

    python -m src.filters prepare --task nsfw|watermark [--sets a,b]     (on Kaggle; internet)

Each task has several community sets (BENCHMARKS, verified on the Hub on 2026-11-06 and pinned to a commit); each
is sampled with a fixed seed, at most PER_CLASS images per class, and mapped to a binary label. A set that cannot be
loaded (a gated one without access) is skipped with a message.

- nsfw: deepghs (deepghs/nsfw_detect, 5 classes, the team's first benchmark), nsfw_xl (markandey14/
  nsfw_detection_xl), unsafebench (yiting/UnsafeBench, real LAION-5B images only; gated);
- watermark: boomb0om (boomb0om/watermarks-validation), scenery (qwertyforce/scenery_watermarks, real marks; the
  training set of `efficientnet`), wm_or_not (prithivMLmods/Watermark-or-Not-20K, synthetic text marks).

Images are copied to work/benchmarks/<task>/<set>/<class>/<file>, labels go
to work/benchmarks/<task>/labels.jsonl as {"image_id": "<set>/<class>/<file stem>", "label": 0 or 1, "file"}, so
results can be split by set and class. Scores go with the other outputs (work/filters/<task>/benchmark/, base.py).
The NSFW sets hold pornographic images: `prepare` refuses to run outside Kaggle unless allowed explicitly, and on
Kaggle the work folder should be outside /kaggle/working, so the images are never saved.
"""

import csv
import io
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..common.io import iter_jsonl, write_jsonl
from ..paths import WORK_DIR

SEED = 42


@dataclass(frozen=True)
class BenchRef:
    image_id: str
    path: Path


@dataclass(frozen=True)
class Row:
    cls: str                # class name in the source set
    label: int              # 1 = positive (NSFW, watermark)
    name: str               # file name to write
    data: bytes | None = None
    src: Path | None = None


def bench_dir(task: str) -> Path:
    return WORK_DIR / "benchmarks" / task


def labels_path(task: str) -> Path:
    return bench_dir(task) / "labels.jsonl"


def read_labels(task: str, sets: list[str] | None = None) -> dict[str, int]:
    return {row["image_id"]: int(row["label"]) for row in iter_jsonl(labels_path(task))
            if sets is None or row["image_id"].split("/")[0] in sets}


def benchmark_refs(task: str, sets: list[str] | None = None) -> list[BenchRef]:
    rows = [row for row in iter_jsonl(labels_path(task)) if sets is None or row["image_id"].split("/")[0] in sets]
    if not rows:
        raise SystemExit(f"no benchmark set for {task}: run `python -m src.filters prepare --task {task}` first")
    return [BenchRef(row["image_id"], bench_dir(task) / row["file"]) for row in rows]


def _token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def sample(rows: list[Row], per_class: int) -> list[Row]:
    """At most `per_class` rows of every class, drawn with SEED from rows sorted by name (reproducible)."""
    rng, out = random.Random(SEED), []
    for cls in sorted({row.cls for row in rows}):
        group = sorted((row for row in rows if row.cls == cls), key=lambda row: row.name)
        out += group if len(group) <= per_class else rng.sample(group, per_class)
    return out


# --- NSFW sets -------------------------------------------------------------------------------------------------

def deepghs_nsfw(tmp: Path) -> list[Row]:
    """deepghs/nsfw_detect, nsfw_dataset_v1.zip: folders drawings, hentai, neutral, porn, sexy. 400 per class,
    drawn exactly as in the team's first benchmark (one random.Random(42) over the sorted classes)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("deepghs/nsfw_detect", "nsfw_dataset_v1.zip", repo_type="dataset", token=_token())
    with zipfile.ZipFile(path) as archive:
        archive.extractall(tmp)
    root = next(folder.parent for folder in tmp.rglob("neutral") if folder.is_dir())
    rng, rows = random.Random(SEED), []
    for cls in ("drawings", "hentai", "neutral", "porn", "sexy"):
        for image_path in rng.sample(sorted((root / cls).glob("*.jpg")), 400):
            rows.append(Row(cls, int(cls in ("porn", "hentai", "sexy")), image_path.name, src=image_path))
    return rows


# --- watermark sets ----------------------------------------------------------------------------------------------

def boomb0om_watermarks(tmp: Path) -> list[Row]:
    """boomb0om/watermarks-validation at a pinned revision: 122 images (62 watermark, 60 clean)."""
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download("boomb0om/watermarks-validation", repo_type="dataset",
                                  revision="d2d22701e18e96abd28fc476e9396501adf87123",
                                  allow_patterns=["clean/**", "watermark/**", "validation_data.csv"]))
    rows = []
    with open(root / "validation_data.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            positive = row["watermark"].strip().lower() in ("true", "1")
            image_path = root / row["path"]
            rows.append(Row("watermark" if positive else "clean", int(positive), image_path.name, src=image_path))
    return rows


def sample_indices(classes: list[str], per_class: int) -> list[int]:
    """Row indices: at most `per_class` of every class, drawn with SEED (rows in their stored order)."""
    rng, out = random.Random(SEED), []
    for cls in sorted(set(classes)):
        indices = [index for index, name in enumerate(classes) if name == cls]
        out += indices if len(indices) <= per_class else sorted(rng.sample(indices, per_class))
    return out


def parquet_rows(repo: str, revision: str, files: list[str], class_of, per_class: int,
                 columns: list[str]) -> list[Row]:
    """Sampled rows of parquet shards of a Hugging Face dataset. `class_of(record)` maps the `columns` of a row to
    (class, label), or None to leave the row out; image bytes are read only for the sampled rows."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    tables = [pq.read_table(hf_hub_download(repo, file, repo_type="dataset", revision=revision, token=_token()))
              for file in files]
    meta = [(table, index, class_of({column: table.column(column)[index].as_py() for column in columns}))
            for table in tables for index in range(table.num_rows)]
    meta = [item for item in meta if item[2] is not None]
    rows = []
    for position in sample_indices([item[2][0] for item in meta], per_class):
        table, index, (cls, label) = meta[position]
        image = table.column("image")[index].as_py()
        rows.append(Row(cls, label, f"{position:06d}.jpg", data=image["bytes"] if isinstance(image, dict) else image))
    return rows


def nsfw_detection_xl(tmp: Path) -> list[Row]:
    """markandey14/nsfw_detection_xl (MIT; source not stated; mixes several sources, the nsfw side includes drawn
    and anime images): one zip with folders nsfw/ (26,799) and sfw/ (21,198)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("markandey14/nsfw_detection_xl", "nsfw_detection_xl.zip", repo_type="dataset",
                           revision="c324522ed1259218acae6f4e4de7e4a34aef026d", token=_token())
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.split("/")[-2:-1] in (["nsfw"], ["sfw"])
                 and name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        names.sort()
        rows = []
        for position in sample_indices([name.split("/")[-2] for name in names], PER_CLASS):
            cls = names[position].split("/")[-2]
            rows.append(Row(cls, int(cls == "nsfw"), f"{position:06d}.jpg", data=archive.read(names[position])))
    return rows


def unsafebench(tmp: Path) -> list[Row]:
    """yiting/UnsafeBench (gated: accept its terms on Hugging Face with the HF_TOKEN account). Only the real images
    (source LAION-5B): positive = unsafe and category Sexual, negative = safe."""
    def class_of(row):
        if "laion" not in str(row["source"]).lower():
            return None
        if str(row["safety_label"]).lower().startswith("safe"):
            return "safe", 0
        return ("sexual", 1) if "sexual" in str(row["category"]).lower() else None
    return parquet_rows("yiting/UnsafeBench", "9f4560ae90059237eb5eafc6bd8108c78639d180",
                        ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"],
                        class_of, PER_CLASS, ["source", "safety_label", "category"])


def scenery_watermarks(tmp: Path) -> list[Row]:
    """qwertyforce/scenery_watermarks (CC BY-NC 4.0): real photos from r/EarthPorn with photographers' marks,
    hand-labelled. The training set of the `efficientnet` detector (qwertyforce/watermark_detection), so its
    score here is not a fair test. Shard 1 of 3 holds both classes (5,191 clean, 2,396 watermark)."""
    names = {0: "clean", 1: "watermark"}
    return parquet_rows("qwertyforce/scenery_watermarks", "4751fb84bfebdb4e79e4fda756602104c23bbc2d",
                        ["data/train-00001-of-00003-54a652031b967a2d.parquet"],
                        lambda row: (names[row["label"]], row["label"]), PER_CLASS, ["label"])


def watermark_or_not(tmp: Path) -> list[Row]:
    """prithivMLmods/Watermark-or-Not-20K (Apache-2.0): stock-style photos with synthetic text marks (probably the
    Kaggle set felicepollano/watermarked-not-watermarked-images); some marks are very faint. Rows are sorted by
    label: shard 0000 is clean, shard 0005 is watermarked."""
    names = {0: "clean", 1: "watermark"}
    return parquet_rows("prithivMLmods/Watermark-or-Not-20K", "f2dbc82ec589b8bc4eb675221a3fb80fd1d95eed",
                        ["dataset/0000.parquet", "dataset/0005.parquet"],
                        lambda row: (names[row["label"]], row["label"]), PER_CLASS, ["label"])


PER_CLASS = 1000
# name -> (task, loader, images per class); the order is the order of the report
BENCHMARKS = {
    "deepghs": ("nsfw", deepghs_nsfw, 400),
    "nsfw_xl": ("nsfw", nsfw_detection_xl, PER_CLASS),
    "unsafebench": ("nsfw", unsafebench, PER_CLASS),
    "boomb0om": ("watermark", boomb0om_watermarks, PER_CLASS),
    "scenery": ("watermark", scenery_watermarks, PER_CLASS),
    "wm_or_not": ("watermark", watermark_or_not, PER_CLASS),
}


def prepare(task: str, allow_local: bool = False, sets: list[str] | None = None) -> dict:
    if not (os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or allow_local):
        raise SystemExit("prepare downloads public benchmark sets (the NSFW ones are pornographic): run it on Kaggle")
    known = [name for name, (set_task, _, _) in BENCHMARKS.items() if set_task == task]
    names = sets or known
    unknown = [name for name in names if name not in known]
    if unknown:
        raise SystemExit(f"unknown {task} sets {unknown}; known: {known}")
    root = bench_dir(task)
    root.mkdir(parents=True, exist_ok=True)
    kept = [row for row in iter_jsonl(labels_path(task)) if row["image_id"].split("/")[0] not in names]
    summary = {}
    for name in names:
        _, loader, per_class = BENCHMARKS[name]
        shutil.rmtree(root / name, ignore_errors=True)
        tmp = root / f"_tmp_{name}"
        tmp.mkdir(parents=True, exist_ok=True)
        print(f"{name}: downloading and sampling", flush=True)
        try:
            rows = sample(loader(tmp), per_class)
        except Exception as error:                       # noqa: BLE001  e.g. a gated set without access
            shutil.rmtree(tmp, ignore_errors=True)
            summary[name] = {"skipped": f"{type(error).__name__}: {error}"[:300]}
            print(f"{name}: skipped ({summary[name]['skipped']})", flush=True)
            continue
        labels = []
        for row in rows:
            file = f"{name}/{row.cls}/{row.name}"
            (root / name / row.cls).mkdir(parents=True, exist_ok=True)
            if row.data is not None:
                from PIL import Image
                with Image.open(io.BytesIO(row.data)) as image:
                    image.convert("RGB").save(root / file, quality=95)
            else:
                shutil.copyfile(row.src, root / file)
            labels.append({"image_id": f"{name}/{row.cls}/{Path(row.name).stem}", "label": row.label, "file": file})
        shutil.rmtree(tmp, ignore_errors=True)
        if len({label["image_id"] for label in labels}) != len(labels):
            raise SystemExit(f"{name}: duplicate image ids")
        kept += labels
        summary[name] = {"images": len(labels), "positive": sum(label["label"] for label in labels)}
        print(f"{name}: {summary[name]['images']} images, {summary[name]['positive']} positive", flush=True)
    random.Random(SEED).shuffle(kept)                    # mixed sets and classes, so a --limit run sees all of them
    write_jsonl(labels_path(task), kept)
    return {"task": task, "sets": summary, "images": len(kept)}
