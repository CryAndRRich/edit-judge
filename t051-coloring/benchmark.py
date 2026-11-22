import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from statistics import mean
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat


SAMPLE = json.loads(Path(__file__).with_name("sample.json").read_text(encoding="utf-8"))
MODELS = {
    "instruct-pix2pix": ("timbrooks/instruct-pix2pix", "31519b5cb02a7fd89b906d88731cd4d6a7bbf88d"),
    "magicbrush": ("vinesmsuic/magicbrush-jul7", "d75a9f18439f67dbdc0ab113036f491ce2256c53"),
}
SETTINGS = {
    "seed": 42,
    "steps": 30,
    "max_side": 512,
    "guidance_scale": 7.5,
    "image_guidance_scale": 1.5,
}
RUNTIME_FIELDS = (
    "image", "status", "width", "height", "seconds", "peak_allocated_mib",
    "peak_reserved_mib", "output", "error",
)
METRICS = ("ssim_luma", "delta_red_minus_blue", "delta_luma")
COMPARE_FIELDS = ("image", "model", "width", "height", *METRICS, "output_sha256")


def load_image(source):
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def sample_paths(root):
    images = {}
    for name, expected_hash in SAMPLE["images"].items():
        direct = root / name
        matches = [direct] if direct.is_file() else [p for p in root.rglob(name) if p.is_file()]
        for path in sorted(matches):
            if hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash:
                images[name] = path
                break
        else:
            raise ValueError(f"{name}: missing or SHA-256 differs from sample.json under {root}")
    return images


def preprocess(image):
    image.thumbnail((SETTINGS["max_side"],) * 2, Image.Resampling.LANCZOS)
    dimensions = tuple(max(8, side // 8 * 8) for side in image.size)
    return image.resize(dimensions, Image.Resampling.LANCZOS)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args, inputs):
    import torch
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required for run.")
    if args.checkpoint and not args.checkpoint.is_dir():
        raise ValueError(f"Local checkpoint directory does not exist: {args.checkpoint}")

    repo, revision = MODELS[args.model]
    args.output.mkdir(parents=True, exist_ok=False)
    image_dir = args.output / "images"
    image_dir.mkdir()
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "repo": repo,
        "revision": args.checkpoint_revision if args.checkpoint else revision,
        "model_source": "local_snapshot" if args.checkpoint else "huggingface",
        "dataset": SAMPLE["dataset"],
        "prompt": SAMPLE["prompt"],
        **SETTINGS,
        "scheduler": "EulerAncestralDiscreteScheduler",
        "dtype": "float16",
        "generator_device": "cpu",
        "cpu_offload": True,
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("torch", "diffusers", "transformers", "accelerate", "huggingface-hub", "Pillow")
        },
        "status": "loading",
    }
    metadata_path = args.output / "run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    rows = []
    try:
        start = time.perf_counter()
        options = {
            "torch_dtype": torch.float16,
            "use_safetensors": args.model == "instruct-pix2pix",
        }
        if not args.checkpoint:
            options["revision"] = revision
            options["token"] = False
        if args.model == "instruct-pix2pix" and (
            not args.checkpoint
            or (args.checkpoint / "unet/diffusion_pytorch_model.fp16.safetensors").is_file()
        ):
            options["variant"] = "fp16"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            str(args.checkpoint.resolve()) if args.checkpoint else repo, **options
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        metadata["load_seconds"] = round(time.perf_counter() - start, 3)
        metadata["status"] = "running"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for name, path in inputs.items():
            row = {"image": name}
            rows.append(row)
            try:
                image = preprocess(load_image(path))
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                output = pipe(
                    prompt=SAMPLE["prompt"],
                    image=image,
                    num_inference_steps=SETTINGS["steps"],
                    guidance_scale=SETTINGS["guidance_scale"],
                    image_guidance_scale=SETTINGS["image_guidance_scale"],
                    generator=torch.Generator(device="cpu").manual_seed(SETTINGS["seed"]),
                )
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                if output.nsfw_content_detected and output.nsfw_content_detected[0]:
                    raise RuntimeError("Output flagged by the checkpoint safety checker")
                edited = output.images[0]
                if not any(low < high for low, high in edited.getextrema()):
                    raise RuntimeError("The model returned a spatially constant image")
                edited_path = image_dir / f"{path.stem}.png"
                edited.save(edited_path)
                row.update(
                    status="ok", width=edited.width, height=edited.height,
                    seconds=round(seconds, 3),
                    peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                    peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 1),
                    output=edited_path.relative_to(args.output).as_posix(),
                )
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                torch.cuda.empty_cache()
            write_csv(args.output / "run.csv", RUNTIME_FIELDS, rows)
            print(f"{args.model} {name}: {row['status']} {row.get('seconds', row.get('error', ''))}", flush=True)
        metadata["status"] = "ok" if all(row["status"] == "ok" for row in rows) else "incomplete"
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return int(metadata["status"] != "ok")


def edited_bytes(source, name):
    if isinstance(source, ZipFile):
        matches = [
            entry for entry in source.infolist()
            if not entry.is_dir() and PurePosixPath(entry.filename).name == name
        ]
        if len(matches) != 1:
            raise ValueError(f"{source.filename}: expected one {name}, found {len(matches)}")
        return source.read(matches[0])
    return (source / name).read_bytes()


def image_metrics(original, edited):
    import numpy as np
    from skimage.metrics import structural_similarity

    original = original.resize(edited.size, Image.Resampling.LANCZOS)
    original_luma, edited_luma = original.convert("L"), edited.convert("L")
    original_r, _, original_b = ImageStat.Stat(original).mean
    edited_r, _, edited_b = ImageStat.Stat(edited).mean
    return {
        "ssim_luma": structural_similarity(np.asarray(original_luma), np.asarray(edited_luma), data_range=255),
        "delta_red_minus_blue": (edited_r - edited_b) - (original_r - original_b),
        "delta_luma": ImageStat.Stat(edited_luma).mean[0] - ImageStat.Stat(original_luma).mean[0],
    }


def comparison_image(path, rows, labels):
    tile_width, tile_height = 256, 260
    canvas = Image.new("RGB", (tile_width * len(labels), tile_height * len(rows)), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    for row_index, (name, images) in enumerate(rows.items()):
        for column, (label, image) in enumerate(zip(labels, images, strict=True)):
            x, y = column * tile_width, row_index * tile_height
            draw.text((x + 12, y + 10), f"{Path(name).stem} / {label}", fill="#182230", font=font)
            tile = ImageOps.contain(image, (224, 224), Image.Resampling.LANCZOS)
            canvas.paste(tile, (x + (tile_width - tile.width) // 2, y + 29))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def compare(args, inputs):
    edits = dict(args.edit)
    if len(edits) != len(args.edit):
        raise ValueError("Model names supplied via --edit must be unique.")
    rows, pictures = [], {}
    with ExitStack() as stack:
        sources = {
            label: stack.enter_context(ZipFile(path)) if path.suffix.lower() == ".zip" else path
            for label, path in edits.items()
        }
        for name, path in inputs.items():
            original = load_image(path)
            pictures[name] = [original]
            for label, source in sources.items():
                raw = edited_bytes(source, f"{path.stem}.png")
                edited = load_image(io.BytesIO(raw))
                pictures[name].append(edited)
                rows.append({
                    "image": name, "model": label, "width": edited.width, "height": edited.height,
                    **image_metrics(original, edited),
                    "output_sha256": hashlib.sha256(raw).hexdigest(),
                })
    summary = []
    for label in edits:
        selected = [row for row in rows if row["model"] == label]
        summary.append({
            "model": label, "images": len(selected),
            **{field: round(mean(row[field] for row in selected), 6) for field in METRICS},
        })
    for row in rows:
        for field in METRICS:
            row[field] = round(row[field], 6)
    write_csv(args.report_dir / "comparison.csv", COMPARE_FIELDS, rows)
    write_csv(args.report_dir / "summary.csv", ("model", "images", *METRICS), summary)
    comparison_image(args.preview, pictures, ["Original", *edits])
    print(f"Compared {len(rows)} outputs. Preview: {args.preview}")
    return 0


def named_edit(value):
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("Expected MODEL=PATH.")
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser(description="T-051 EditJudge sunset coloring benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("run", help="Run inference for one model on all five images")
    generate.add_argument("--images", type=Path, required=True)
    generate.add_argument("--model", choices=MODELS, required=True)
    generate.add_argument("--output", type=Path, required=True, help="New directory for this run")
    generate.add_argument("--checkpoint", type=Path, help="Optional local Diffusers snapshot")
    generate.add_argument("--checkpoint-revision", help="Source commit of a local snapshot, if known")
    evaluate = commands.add_parser("compare", help="Compare image folders or ZIP archives")
    evaluate.add_argument("--images", type=Path, required=True)
    evaluate.add_argument("--edit", type=named_edit, action="append", required=True, metavar="MODEL=PATH")
    evaluate.add_argument("--report-dir", type=Path, required=True)
    evaluate.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run" and args.checkpoint_revision and not args.checkpoint:
        parser.error("--checkpoint-revision requires --checkpoint")
    try:
        inputs = sample_paths(args.images)
        return (run if args.command == "run" else compare)(args, inputs)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"T-051: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
