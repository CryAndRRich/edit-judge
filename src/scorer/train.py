"""Train a scoring head on cached embeddings (configs/scorer.toml), and compare every run with the baselines.

Features of an edit e of a sample with original o and request t, per backbone (vectors L2-normalised):
- "vec": e, e - o and, for backbones with text, e * t (element-wise): what the edit looks like, what changed, and
  how the edit relates to the request;
- "sim": cos(e, o), and with text cos(e, t), cos(o, t), cos(e, t) - cos(o, t), cos(e - o, t), plus |e - o|.
Features are standardised with the mean and deviation of the training edits.

Training: pairs of the train split, model selection (early stopping) on dev pairwise accuracy, test reported once.
Writes data/scorer/runs/<name>/: metrics.json (train, dev, test), scores.jsonl (every edit), head.pt (weights,
feature standardisation and config, enough to score new edits from the same backbones).
"""

import copy
import glob
import json
import random
import time
import tomllib

import numpy as np

from ..common.device import pick_device
from ..common.io import write_json, write_jsonl
from ..paths import REPO_ROOT, SCORER_DIR
from .backbones import BACKBONES
from .evaluate import eval_data, evaluate_scores
from .embed import load_embeddings

INPUTS = ("vec", "sim")
DEFAULTS = {"head": "mlp", "loss": "soft", "inputs": ["vec", "sim"], "hidden": 256, "dropout": 0.2, "lr": 1e-3,
            "weight_decay": 1e-2, "batch_size": 256, "epochs": 100, "patience": 10, "seed": 2026}


def load_config(path: str = "configs/scorer.toml") -> dict:
    with open(REPO_ROOT / path if not path.startswith("/") else path, "rb") as fh:
        return tomllib.load(fh)


def _rows_cos(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return np.where(n > 0, (x * y).sum(axis=1) / np.maximum(n, 1e-12), 0.0)


def edit_features(edits: list[tuple[str, str]], backbones: list[str], inputs: list[str]) -> tuple[np.ndarray, list[str]]:
    """Feature matrix of the edits (sample ID, key) that every backbone embedded, and their image_ids."""
    for b in backbones:
        if b not in BACKBONES:
            raise SystemExit(f"no backbone {b!r}; registered: {sorted(BACKBONES)}")
    unknown = set(inputs) - set(INPUTS)
    if unknown:
        raise SystemExit(f"unknown inputs {sorted(unknown)}; expected some of {INPUTS}")
    loaded = {b: (load_embeddings(b, "images"), load_embeddings(b, "texts")) for b in backbones}
    keep = [(s, k) for s, k in edits
            if all(f"{s}/{k}" in im and f"{s}/original" in im for im, _ in loaded.values())]
    blocks = []
    for b in backbones:
        images, texts = loaded[b]
        if not images:
            raise SystemExit(f"no embeddings for {b}: run `python -m src.scorer embed --backbone {b}` first")
        e = np.stack([images[f"{s}/{k}"] for s, k in keep])
        o = np.stack([images[f"{s}/original"] for s, k in keep])
        t = np.stack([texts[s] for s, k in keep]) if texts else None
        if "vec" in inputs:
            blocks += [e, e - o] + ([e * t] if t is not None else [])
        if "sim" in inputs:
            sims = [_rows_cos(e, o), np.linalg.norm(e - o, axis=1)]
            if t is not None:
                et, ot = _rows_cos(e, t), _rows_cos(o, t)
                sims += [et, ot, et - ot, _rows_cos(e - o, t)]
            blocks.append(np.stack(sims, axis=1))
    return np.concatenate(blocks, axis=1).astype(np.float32), [f"{s}/{k}" for s, k in keep]


def train_run(run: dict, defaults: dict | None = None, version: str = "v1", device: str | None = None) -> dict:
    """Train one head (`run`: name, backbones and any DEFAULTS key), save it and return its metrics."""
    import torch

    from .heads import ScoreHead, pair_loss

    cfg = {**DEFAULTS, **(defaults or {}), **run}
    device = pick_device(device)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    pairs, rankings = eval_data(version)
    edits = sorted({(p["id"], k) for ps in pairs.values() for p in ps for k in (p["a"], p["b"])})
    features, ids = edit_features(edits, cfg["backbones"], cfg["inputs"])
    row = {image_id: i for i, image_id in enumerate(ids)}

    def index(split):
        ps = [p for p in pairs.get(split, []) if f"{p['id']}/{p['a']}" in row and f"{p['id']}/{p['b']}" in row]
        ia = torch.tensor([row[f"{p['id']}/{p['a']}"] for p in ps])
        ib = torch.tensor([row[f"{p['id']}/{p['b']}"] for p in ps])
        hard = torch.tensor([float(p["winner"] == p["a"]) for p in ps])
        soft = torch.tensor([float(p["p_a"]) for p in ps])
        return ia, ib, (hard if cfg["loss"] == "bt" else soft), hard

    train, dev = index("train"), index("dev")
    if not len(train[0]) or not len(dev[0]):
        raise SystemExit("no train or dev pairs with embeddings")
    train_rows = torch.unique(torch.cat(train[:2]))
    x = torch.from_numpy(features)
    mean, std = x[train_rows].mean(0), x[train_rows].std(0).clamp_min(1e-6)
    x = ((x - mean) / std).to(device)

    model = ScoreHead(x.shape[1], cfg["head"], cfg["hidden"], cfg["dropout"], cfg["loss"] == "uncertainty").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    def scores():
        model.eval()
        with torch.no_grad():
            return model(x)[0]

    def accuracy(s, ia, ib, hard):
        sa, sb = s[ia], s[ib]
        return float(torch.where(sa == sb, 0.5, ((sa > sb).float() == hard.to(device)).float()).mean())

    started, best, best_acc, best_epoch, history = time.monotonic(), None, -1.0, 0, []
    ia, ib, target, _ = train
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order, total = torch.randperm(len(ia)), 0.0
        for i in range(0, len(ia), cfg["batch_size"]):
            j = order[i:i + cfg["batch_size"]]
            mu_a, lv_a = model(x[ia[j].to(device)])
            mu_b, lv_b = model(x[ib[j].to(device)])
            loss = pair_loss(mu_a, mu_b, lv_a, lv_b, target[j].to(device), cfg["loss"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(j)
        s = scores()
        dev_acc = accuracy(s, dev[0], dev[1], dev[3])
        history.append({"epoch": epoch, "train_loss": round(total / len(ia), 5),
                        "train_accuracy": round(accuracy(s, train[0], train[1], train[3]), 4),
                        "dev_accuracy": round(dev_acc, 4)})
        if dev_acc > best_acc:
            best, best_acc, best_epoch = copy.deepcopy(model.state_dict()), dev_acc, epoch
        elif epoch - best_epoch >= cfg["patience"]:
            break
    model.load_state_dict(best)
    final = scores().cpu().tolist()
    all_scores = dict(zip(ids, final))

    out = SCORER_DIR / "runs" / cfg["name"]
    metrics = {"name": cfg["name"], "config": cfg, "version": version, "device": device, "features": x.shape[1],
               "edits": len(ids), "best_epoch": best_epoch, "seconds": round(time.monotonic() - started, 1),
               **evaluate_scores(all_scores, pairs, rankings, splits=("train", "dev", "test")), "history": history}
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "scores.jsonl", ({"image_id": i, "score": round(s, 5)} for i, s in zip(ids, final)))
    torch.save({"config": cfg, "state_dict": model.state_dict(), "mean": mean, "std": std}, out / "head.pt")
    return {k: metrics[k] for k in ("name", "features", "edits", "best_epoch", "seconds")} | {
        split: {k: metrics[split][k] for k in ("pairs", "accuracy", "ci95", "top1")} for split in ("dev", "test")}


def report() -> dict:
    """One table of every baseline and run: dev and test accuracy (test CI, confident pairs, top-1, by source),
    written to data/scorer/report.md and report.json."""
    rows, human = [], {}
    baselines_path = SCORER_DIR / "baselines.json"
    if baselines_path.exists():
        b = json.loads(baselines_path.read_text(encoding="utf-8"))
        human = b["human"]
        rows += [{"name": name, "kind": "baseline", **m} for name, m in b["baselines"].items()]
    for path in sorted(glob.glob(str(SCORER_DIR / "runs" / "*" / "metrics.json"))):
        m = json.loads(open(path, encoding="utf-8").read())
        rows.append({"name": m["name"], "kind": "trained", "dev": m["dev"], "test": m["test"]})
    rows.sort(key=lambda r: (r["kind"], -(r["dev"]["accuracy"] or 0)))
    lines = ["| model | kind | dev acc | test acc (95% CI) | test weighted | test confident | test top-1 | "
             "test RealEdit | test GIER |", "|" + "---|" * 9]
    for r in rows:
        d, t = r["dev"], r["test"]
        src = lambda s: (t["by_source"].get(s) or {}).get("accuracy")
        lines.append(f"| {r['name']} | {r['kind']} | {d['accuracy']} | {t['accuracy']} {t['ci95']} | "
                     f"{t['weighted_accuracy']} | {t['confident_accuracy']} | {t['top1']} (chance {t['top1_chance']}) | "
                     f"{src('realedit')} | {src('gier')} |")
    if human:
        h = human.get("test", {})
        lines += ["", f"Human ceiling (test): one annotator agrees with the other two (when they agree) on "
                      f"{h.get('annotator_vs_other_two')} of pairs, with the final winner on {h.get('annotator_vs_final')}."]
    (SCORER_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"human": human, "rows": [{"name": r["name"], "kind": r["kind"], "dev": r["dev"]["accuracy"],
                                         "test": r["test"]["accuracy"], "test_ci95": r["test"]["ci95"],
                                         "test_top1": r["test"]["top1"]} for r in rows]}
    write_json(SCORER_DIR / "report.json", result)
    return result
