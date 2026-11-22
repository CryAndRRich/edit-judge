"""Charts, a table and image grids for the Enhance benchmark (MODEL_LIST.md, list 4).

    python scripts/visualize_enhance.py [--editors work/editors/full] [--out reports/enhance] [--grid-dir <dir>]

Reads <editors>/<model>/runs.jsonl (time, peak GPU memory) and <editors>/scores/<reward>.jsonl (rows
{"image_id": "<model>/<sample id>", "score"}, from scoring/score_runner.py; "original" is the unedited input).
Writes into --out (no photos, can be committed): comparison.png (time, peak GPU memory, and the paired score gain
over the unedited input for each reward model), stats.json, stats.md. The image grids show photos of people, so they
go to --grid-dir (default <editors>/report, under the git-ignored work/) and are never to be published.

Colours: the first four slots of the validated categorical palette, in a fixed model order (a model keeps its colour
in every chart). Its light-mode contrast is below 3:1 for two slots, so every bar carries its value as a label.
"""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from PIL import Image, ImageDraw, ImageFont          # noqa: E402

MODELS = [("realesrgan", "Real-ESRGAN x4plus", "#2a78d6"), ("hvi_cidnet", "HVI-CIDNet", "#eb6834"),
          ("instructir", "InstructIR", "#1baf7a"), ("vosr", "VOSR 2.0", "#eda100")]
REWARDS = [("imagereward", "ImageReward"), ("editreward", "EditReward")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2dd", "#fcfcfb"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def bootstrap_ci(values: list[float], draws: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def collect(editors: Path) -> dict:
    stats = {}
    for name, label, _ in MODELS:
        rows = read_jsonl(editors / name / "runs.jsonl")
        ok = [r for r in rows if r["status"] == "ok"]
        later = [r["seconds"] for r in ok if not r.get("first_image")] or [r["seconds"] for r in ok]
        stats[name] = {
            "label": label, "samples": len(rows), "ok": len(ok),
            "failed": sum(r["status"] == "failed" for r in rows), "oom": sum(bool(r.get("oom")) for r in rows),
            "skipped": sum(r["status"] == "skipped" for r in rows),
            "black": sum(bool(r.get("black")) for r in ok), "nonfinite": sum(bool(r.get("nonfinite")) for r in ok),
            "seconds_mean": sum(later) / len(later) if later else None,
            "seconds_first": next((r["seconds"] for r in ok if r.get("first_image")), None),
            "peak_gpu_gb": max((r["peak_gpu_gb"] for r in ok if r.get("peak_gpu_gb") is not None), default=None),
        }
        meta = editors / name / "meta.json"
        if meta.exists():
            stats[name]["load_seconds"] = json.loads(meta.read_text()).get("load_seconds")
        for reward, _ in REWARDS:
            scores = {row["image_id"]: row["score"] for row in read_jsonl(editors / "scores" / f"{reward}.jsonl")
                      if row.get("score") is not None}
            pairs = [(scores[f"{name}/{r['sample_id']}"], scores[f"original/{r['sample_id']}"]) for r in ok
                     if f"{name}/{r['sample_id']}" in scores and f"original/{r['sample_id']}" in scores]
            if pairs:
                gains = [edit - base for edit, base in pairs]
                low, high = bootstrap_ci(gains)
                stats[name][reward] = {"n": len(pairs), "mean": sum(e for e, _ in pairs) / len(pairs),
                                       "original_mean": sum(b for _, b in pairs) / len(pairs),
                                       "gain": sum(gains) / len(gains), "gain_ci": [low, high],
                                       "win_rate": sum(g > 0 for g in gains) / len(gains)}
    return stats


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def bars(ax, title: str, values: list[tuple[str, str, float | None, tuple | None]], fmt: str, zero_line=False):
    style(ax, title)
    for position, (label, colour, value, interval) in enumerate(reversed(values)):
        if value is None:
            ax.text(0, position, "  no data", va="center", color=MUTED, fontsize=9)
            continue
        ax.barh(position, value, height=0.5, color=colour, edgecolor=SURFACE, linewidth=2)
        if interval:
            ax.plot(interval, [position, position], color=INK, linewidth=1.2)
        edge = max(value, interval[1] if interval else value, 0)
        ax.text(edge, position, "  " + format(value, fmt), va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(values)), [label for label, *_ in reversed(values)], color=INK)
    if zero_line:
        ax.axvline(0, color=MUTED, linewidth=1)
    ax.margins(x=0.25)


def draw_charts(stats: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), facecolor=SURFACE)
    bars(axes[0][0], "Mean time per image (s, T4)",
         [(label, c, stats[n]["seconds_mean"], None) for n, label, c in MODELS], ".2f")
    bars(axes[0][1], "Peak GPU memory (GB)",
         [(label, c, stats[n]["peak_gpu_gb"], None) for n, label, c in MODELS], ".2f")
    for axis, (reward, name) in zip((axes[1][0], axes[1][1]), REWARDS):
        entries = [(f"{label} (n={stats[n][reward]['n']})" if reward in stats[n] else label, c,
                    stats[n][reward]["gain"] if reward in stats[n] else None,
                    tuple(stats[n][reward]["gain_ci"]) if reward in stats[n] else None) for n, label, c in MODELS]
        bars(axis, f"{name}: gain over the input", entries, "+.3f", zero_line=True)
    fig.text(0.01, 0.005, "Same dev-split Enhance samples for every model, except HVI-CIDNet (low-light instructions only). Gain =\n"
             "score(output) - score(input), paired per sample; bars show the mean, the line its 95% bootstrap CI.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "comparison.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def stats_table(stats: dict) -> str:
    head = ("| Model | ok / failed / skipped | black / non-finite | s per image | first image (s) | load (s) | peak GPU (GB) |"
            + "".join(f" {name} gain [95% CI], win rate |" for _, name in REWARDS))
    lines = [head, "|" + "---|" * (7 + len(REWARDS))]
    for name, label, _ in MODELS:
        s = stats[name]
        cells = [label, f"{s['ok']} / {s['failed']} / {s['skipped']}", f"{s['black']} / {s['nonfinite']}",
                 "-" if s["seconds_mean"] is None else f"{s['seconds_mean']:.2f}",
                 "-" if s["seconds_first"] is None else f"{s['seconds_first']:.1f}",
                 "-" if s.get("load_seconds") is None else f"{s['load_seconds']:.0f}",
                 "-" if s["peak_gpu_gb"] is None else f"{s['peak_gpu_gb']:.2f}"]
        for reward, _ in REWARDS:
            r = s.get(reward)
            cells.append("-" if not r else f"{r['gain']:+.3f} [{r['gain_ci'][0]:+.3f}, {r['gain_ci'][1]:+.3f}], "
                         f"{r['win_rate']:.0%} (n={r['n']})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def font(size: int):
    for candidate in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_grids(editors: Path, out: Path, rows_per_grid: int = 6, thumb: int = 300) -> list[Path]:
    samples = json.loads((editors / "samples.json").read_text())["samples"]
    ran = {name: {r["sample_id"]: r["status"] for r in read_jsonl(editors / name / "runs.jsonl")}
           for name, *_ in MODELS}
    ordered = [s for s in samples if s["low_light"]] + [s for s in samples if not s["low_light"]]
    columns = [("Input", "_inputs")] + [(label, name) for name, label, _ in MODELS]
    out.mkdir(parents=True, exist_ok=True)
    label_font, small = font(15), font(13)
    paths = []
    for start in range(0, len(ordered), rows_per_grid):
        chunk = ordered[start:start + rows_per_grid]
        sheet = Image.new("RGB", (len(columns) * thumb, 24 + len(chunk) * (thumb + 22)), "white")
        draw = ImageDraw.Draw(sheet)
        for column, (label, _) in enumerate(columns):
            draw.text((column * thumb + 6, 4), label, fill="black", font=label_font)
        for row, sample in enumerate(chunk):
            top = 24 + row * (thumb + 22)
            draw.text((6, top + 2), sample["instruction"][:120] + ("   [low-light]" if sample["low_light"] else ""),
                      fill="#52514e", font=small)
            for column, (_, folder) in enumerate(columns):
                box = (column * thumb, top + 22)
                path = editors / folder / f"{sample['id']}.png"
                if path.exists():
                    image = Image.open(path).convert("RGB")
                    image.thumbnail((thumb - 4, thumb - 4))
                    sheet.paste(image, (box[0] + 2, box[1] + 2))
                else:
                    status = ran.get(folder, {}).get(sample["id"], "no output")
                    draw.rectangle((box[0] + 2, box[1] + 2, box[0] + thumb - 3, box[1] + thumb - 3), fill="#ecebe6")
                    draw.text((box[0] + 12, box[1] + thumb // 2), status, fill="#52514e", font=label_font)
        path = out / f"grid_{start // rows_per_grid + 1}.jpg"
        sheet.save(path, quality=88)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editors", default="work/editors/full")
    parser.add_argument("--out", default="reports/enhance")
    parser.add_argument("--grid-dir")
    args = parser.parse_args()
    editors, out = Path(args.editors), Path(args.out)
    stats = collect(editors)
    draw_charts(stats, out)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    (out / "stats.md").write_text(stats_table(stats) + "\n")
    grids = draw_grids(editors, Path(args.grid_dir) if args.grid_dir else editors / "report")
    print(stats_table(stats))
    print("charts:", out / "comparison.png", "\ngrids (photos of people, local only):", *map(str, grids))


if __name__ == "__main__":
    main()
