"""The report of a task: did each model run, how fast, and how its edits compare with the human ones.

work/editors/<task>/report.md (small files only: runs on a laptop) has
- the plans and masks: answers with problems, masks found, median share of the image masked;
- per model: ok / failed / black / no-mask, median seconds per image, peak GPU memory (from runs.jsonl);
- per model, from scores.jsonl (only samples where the model's output is ok): how often the scorer puts it above
  the best human edit of the final ranking, the mean share of human edits it beats, and how often it is the best
  model of the sample;
and work/editors/<task>/sheets/sheet-NN.jpg, contact sheets for checking by eye: one row per sample with the
original, the mask in red, the best human edit and every model's output. They show real people: team only, never
published.

With about 30 samples per task the shares have wide intervals (about +-18 points at 50%): they show whether a model
works and roughly where it stands, not a ranking of close models.
"""

import statistics

from ..common.dataset import iter_samples
from ..common.io import iter_jsonl
from ..paths import EDITORS_DIR, SAMPLES_DIR
from . import editors_of, get_editor
from .base import load_runs, output_path, summary
from .grounding import load_masks, mask_path
from .plan import DEFAULT_CLIENT, load_plans, plan_version
from .samples import load_task_samples
from .score import load_scores, scores_path, top_human

TILE = 224
ROWS_PER_SHEET = 8


def model_scores(task: str, models: list[str]) -> dict[str, dict]:
    scores = load_scores(task)
    top = {row["id"]: row["top_human"] for row in iter_jsonl(scores_path(task))}
    result = {}
    for name in models:
        beats_top, beats_share, best_model = [], [], []
        for sample_id, edits in scores.items():
            mine = edits.get(f"model:{name}")
            if mine is None:
                continue
            humans = [v for k, v in edits.items() if k.startswith("human:")]
            others = [v for k, v in edits.items() if k.startswith("model:") and k != f"model:{name}"]
            beats_top.append(mine > edits.get(f"human:{top[sample_id]}", float("inf")))
            beats_share.append(sum(mine > v for v in humans) / len(humans))
            best_model.append(all(mine > v for v in others))
        count = len(beats_top)
        result[name] = {"scored": count,
                        "beats_top_human": sum(beats_top) / count if count else None,
                        "beats_humans": statistics.mean(beats_share) if count else None,
                        "best_model": sum(best_model) / count if count else None}
    return result


def percent(value) -> str:
    return "-" if value is None else f"{value:.0%}"


def tile(path, label: str, mask=None):
    """A TILE x TILE thumbnail of `path` with a label; `mask` (path) is drawn over it in red."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (TILE, TILE + 16), (40, 40, 40))
    if path is not None and path.exists():
        with Image.open(path) as im:
            image = im.convert("RGB")
        if mask is not None and mask.exists():
            with Image.open(mask) as m:
                red = Image.new("RGB", image.size, (255, 0, 0))
                image = Image.composite(Image.blend(image, red, 0.5), image, m.convert("L").resize(image.size))
        image.thumbnail((TILE, TILE))
        canvas.paste(image, ((TILE - image.width) // 2, 16 + (TILE - image.height) // 2))
    ImageDraw.Draw(canvas).text((4, 2), label[:34], fill=(255, 255, 255))
    return canvas


def best_human_edits(ids: set[str]) -> dict[str, str]:
    """Sample ID -> the key of its best human edit in the final ranking (from the dataset; empty without it)."""
    return {s["id"]: top_human(s["pairs"]) for s in iter_samples() if s["id"] in ids}


def contact_sheets(task: str, samples: list[dict], models: list[str], version: int) -> list:
    """Sheets of the samples that have a plan; a mask or an output is shown only when it is current (the mask of
    this plan version and grounding rules, an ok output made from it), never an older file left on disk."""
    from PIL import Image

    masks = load_masks(version)
    runs = {name: load_runs(name, version) for name in models}
    samples = [s for s in samples if s["id"] in masks]
    top = best_human_edits({s["id"] for s in samples})
    scores = load_scores(task)
    folder = EDITORS_DIR / task / "sheets"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("sheet-*.jpg"):
        old.unlink()
    columns = 3 + len(models)
    written = []
    for start in range(0, len(samples), ROWS_PER_SHEET):
        chunk = samples[start:start + ROWS_PER_SHEET]
        sheet = Image.new("RGB", (columns * TILE, len(chunk) * (TILE + 16)), (20, 20, 20))
        for row, sample in enumerate(chunk):
            sid = sample["id"]
            original = SAMPLES_DIR / sid / "original.jpg"
            human = top.get(sid)
            tiles = [tile(original, f"{sid} {sample['op']}"),
                     tile(original, f"mask {masks[sid]['status']}",
                          mask_path(sid) if masks[sid]["status"] == "ok" else None),
                     tile(SAMPLES_DIR / sid / f"{human}.jpg" if human else None,
                          f"best human {human or ''} {scores.get(sid, {}).get(f'human:{human}', '')}")]
            for name in models:
                value = scores.get(sid, {}).get(f"model:{name}")
                status = runs[name].get(sid, {}).get("status", "-")
                tiles.append(tile(output_path(name, sid) if status == "ok" else None,
                                  f"{name} {status if status != 'ok' else ''} {'' if value is None else value}"))
            for column, image in enumerate(tiles):
                sheet.paste(image, (column * TILE, row * (TILE + 16)))
        path = folder / f"sheet-{start // ROWS_PER_SHEET + 1:02d}.jpg"
        sheet.save(path, quality=85)
        written.append(path)
    return written


def report(task: str, client_name: str = DEFAULT_CLIENT, version: int | None = None, sheets: bool = True) -> str:
    version = version or plan_version(task)
    samples = load_task_samples(task)
    ids = {s["id"] for s in samples}
    plans = {k: v for k, v in load_plans(client_name, version).items() if k in ids}
    masks = {k: v for k, v in load_masks(version).items() if k in ids}
    models = editors_of(task)
    lines = [f"# Editing models: {task}", "",
             f"{len(samples)} dev samples ({', '.join(sorted({s['op'] for s in samples}))}); plans by {client_name} "
             f"(plan_v{version}); masks by Grounding DINO + SAM 2.1 (add: the plan's box).", "",
             "## Plans and masks", "",
             "| Planned | With problems | Masks ok | No mask | Median share masked |", "|---|---|---|---|---|"]
    coverage = [m["coverage"] for m in masks.values() if m["status"] == "ok"]
    ops = sorted({m["op"] for m in masks.values()})
    lines.append(f"| {len(plans)} | {sum(bool(p['problems']) for p in plans.values())} | "
                 f"{sum(m['status'] == 'ok' for m in masks.values())} | "
                 f"{sum(m['status'] == 'no-mask' for m in masks.values())} | "
                 f"{percent(statistics.median(coverage)) if coverage else '-'} |")
    by_source = {}
    for m in masks.values():
        for source in {b[-1] for b in m["boxes"]} or {"-"}:
            by_source[source] = by_source.get(source, 0) + 1
    lines += ["", "Masks by op: " + ", ".join(f"{op} {sum(m['op'] == op and m['status'] == 'ok' for m in masks.values())}"
                                          f"/{sum(m['op'] == op for m in masks.values())}" for op in ops),
              "Boxes from: " + ", ".join(f"{k} {v}" for k, v in sorted(by_source.items())) +
              " (dino: Grounding DINO inside Qwen's box; qwen: Qwen's box, DINO found nothing there; plan: add)"]
    lines += ["", "## Runs", "",
              "| Model | Input | Ok | Failed | Black | No mask | n/a | Median s / image | Peak GPU GB |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name in models:
        s = summary(name, task, samples, version)
        editor = get_editor(name)
        kind = ("mask" if editor.uses_mask else "instruction") + (f" ({', '.join(editor.ops)})" if editor.ops else "")
        lines.append(f"| {name} | {kind} | {s['ok']} | {s['failed']} | {s['black']} | {s['no-mask']} | {s['n/a']} | "
                     f"{s['median_seconds'] if s['median_seconds'] is not None else '-'} | "
                     f"{s['gpu_peak_gb'] if s['gpu_peak_gb'] is not None else '-'} |")
    failures = [(name, r) for name in models for r in load_runs(name, version).values()
                if r["id"] in ids and r["status"] in ("failed", "black")]
    if failures:
        lines += ["", "Failures (last attempt):", ""]
        lines += [f"- {name} {r['id']}: {r['status']} {r.get('error') or ''}" for name, r in failures[:20]]
    if scores_path(task).exists():
        lines += ["", "## Scorer", "",
                  "| Model | Scored | Above the best human edit | Share of human edits beaten | Best model |",
                  "|---|---|---|---|---|"]
        for name, m in model_scores(task, models).items():
            lines.append(f"| {name} | {m['scored']} | {percent(m['beats_top_human'])} | "
                         f"{percent(m['beats_humans'])} | {percent(m['best_model'])} |")
        lines += ["", "The scorer learned from human edits; check the contact sheets before trusting these shares."]
    if sheets:
        written = contact_sheets(task, samples, models, version)
        lines += ["", "## Contact sheets (team only)", ""] + [f"- `{p.relative_to(EDITORS_DIR)}`" for p in written]
    text = "\n".join(lines) + "\n"
    out = EDITORS_DIR / task / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text
