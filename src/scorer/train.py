"""Train scoring heads on cached embeddings (configs/scorer.toml).

Features of an edit e of a sample with original o and text t (the simplified instruction, or the request as
written), per backbone (vectors L2-normalised):
- "vec": e, e - o and, for backbones with text, e * t (element-wise): what the edit looks like, what changed, and
  how the edit relates to the text;
- "sim": cos(e, o), |e - o| and, with text, cos(e, t), cos(o, t), cos(e, t) - cos(o, t), cos(e - o, t).
Features are standardised with the mean and deviation of the training edits.

Training (pairs of the train split): every run trains one head per seed. Each head keeps the epoch with the
lowest dev loss (dev loss is smoother than dev accuracy on a few thousand pairs), with early stopping. A run's score
of an edit is the mean of its heads' standardised scores (an ensemble over seeds); the report gives the ensemble's
results and the mean and deviation of the single heads, so differences between runs can be told from seed noise.
Writes work/scorer/runs/<name>/: metrics.json, scores.jsonl (ensemble score of every edit), head.pt (every head,
the feature standardisation and the config: enough to score new edits from the same backbones).
"""

import copy
import random
import time
import tomllib

import numpy as np

from ..common.device import pick_device
from ..common.io import write_json, write_jsonl
from ..paths import REPO_ROOT, SCORER_DIR
from .backbones import BACKBONES, get_backbone
from .embed import load_embeddings
from .evaluate import eval_data, evaluate_scores

INPUTS = ("vec", "sim")
DEFAULTS = {"head": "mlp", "loss": "soft", "inputs": ["vec", "sim"], "text": "instruction", "hidden": 64,
            "dropout": 0.3, "input_dropout": 0.1, "lr": 1e-4, "weight_decay": 0.05, "batch_size": 256,
            "epochs": 200, "patience": 8, "seeds": [0, 1, 2, 3, 4]}
SPLITS = ("train", "dev", "test")


def load_config(path: str = "configs/scorer.toml") -> dict:
    with open(REPO_ROOT / path if not path.startswith("/") else path, "rb") as handle:
        return tomllib.load(handle)


def rows_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.where(norms > 0, (first * second).sum(axis=1) / np.maximum(norms, 1e-12), 0.0)


def edit_features(edits: list[tuple[str, str]], backbones: list[str], inputs: list[str],
                  text_kind: str = "instruction") -> tuple[np.ndarray, list[str]]:
    """Feature matrix of the edits (sample ID, key) that every backbone embedded, and their image_ids."""
    for name in backbones:
        if name not in BACKBONES:
            raise SystemExit(f"no backbone {name!r}; registered: {sorted(BACKBONES)}")
    unknown = set(inputs) - set(INPUTS)
    if unknown:
        raise SystemExit(f"unknown inputs {sorted(unknown)}; expected some of {INPUTS}")
    loaded = {}
    for name in backbones:
        images = load_embeddings(name, "images")
        if not images:
            raise SystemExit(f"no embeddings for {name}: run `python -m src.scorer embed` first")
        texts = load_embeddings(name, f"texts-{text_kind}") if get_backbone(name).has_text else {}
        loaded[name] = (images, texts)
    keep = [(sample_id, key) for sample_id, key in edits
            if all(f"{sample_id}/{key}" in images and f"{sample_id}/original" in images
                   and (not texts or sample_id in texts) for images, texts in loaded.values())]
    blocks = []
    for name in backbones:
        images, texts = loaded[name]
        edit = np.stack([images[f"{sample_id}/{key}"] for sample_id, key in keep])
        original = np.stack([images[f"{sample_id}/original"] for sample_id, _ in keep])
        text = np.stack([texts[sample_id] for sample_id, _ in keep]) if texts else None
        if "vec" in inputs:
            blocks += [edit, edit - original] + ([edit * text] if text is not None else [])
        if "sim" in inputs:
            similarities = [rows_cosine(edit, original), np.linalg.norm(edit - original, axis=1)]
            if text is not None:
                edit_text, original_text = rows_cosine(edit, text), rows_cosine(original, text)
                similarities += [edit_text, original_text, edit_text - original_text,
                                 rows_cosine(edit - original, text)]
            blocks.append(np.stack(similarities, axis=1))
    return (np.concatenate(blocks, axis=1).astype(np.float32),
            [f"{sample_id}/{key}" for sample_id, key in keep])


def train_head(cfg: dict, features, train, dev, seed: int, device: str) -> tuple[dict, dict]:
    """One head: the weights of the epoch with the lowest dev loss, and the training history."""
    import torch

    from .heads import ScoreHead, pair_loss

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ScoreHead(features.shape[1], cfg["head"], cfg["hidden"], cfg["dropout"], cfg["input_dropout"],
                      cfg["loss"] == "uncertainty").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    target_key = "hard" if cfg["loss"] == "bt" else "soft"

    def loss_on(pairs) -> torch.Tensor:
        mu_a, logvar_a = model(features[pairs["a"]])
        mu_b, logvar_b = model(features[pairs["b"]])
        return pair_loss(mu_a, mu_b, logvar_a, logvar_b, pairs[target_key], cfg["loss"])

    best_state, best_loss, best_epoch, history = None, float("inf"), 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order, total = torch.randperm(len(train["a"]), device=device), 0.0
        for start in range(0, len(order), cfg["batch_size"]):
            batch = order[start:start + cfg["batch_size"]]
            loss = loss_on({key: values[batch] for key, values in train.items()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            dev_loss = loss_on(dev).item()
        history.append({"epoch": epoch, "train_loss": round(total / len(order), 5), "dev_loss": round(dev_loss, 5)})
        if dev_loss < best_loss:
            best_state, best_loss, best_epoch = copy.deepcopy(model.state_dict()), dev_loss, epoch
        elif epoch - best_epoch >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return {"model": model, "best_epoch": best_epoch, "dev_loss": round(best_loss, 5)}, history


def train_run(run: dict, defaults: dict | None = None, version: str = "all", device: str | None = None) -> dict:
    """Train one run (`run`: name, backbones and any DEFAULTS key) with every seed, save it, return a summary."""
    import torch

    cfg = {**DEFAULTS, **(defaults or {}), **run}
    device = pick_device(device)
    started = time.monotonic()
    pairs, rankings = eval_data(version)
    edits = sorted({(pair["id"], key) for split_pairs in pairs.values() for pair in split_pairs
                    for key in (pair["a"], pair["b"])})
    features, ids = edit_features(edits, cfg["backbones"], cfg["inputs"], cfg["text"])
    row_of = {image_id: index for index, image_id in enumerate(ids)}

    def pair_tensors(split: str) -> dict:
        kept = [pair for pair in pairs.get(split, [])
                if f"{pair['id']}/{pair['a']}" in row_of and f"{pair['id']}/{pair['b']}" in row_of]
        return {"a": torch.tensor([row_of[f"{pair['id']}/{pair['a']}"] for pair in kept], device=device),
                "b": torch.tensor([row_of[f"{pair['id']}/{pair['b']}"] for pair in kept], device=device),
                "hard": torch.tensor([float(pair["winner"] == pair["a"]) for pair in kept], device=device),
                "soft": torch.tensor([float(pair["p_a"]) for pair in kept], device=device)}

    train, dev = pair_tensors("train"), pair_tensors("dev")
    if not len(train["a"]) or not len(dev["a"]):
        raise SystemExit("no train or dev pairs with embeddings")
    matrix = torch.from_numpy(features)
    train_rows = torch.unique(torch.cat([train["a"], train["b"]]).cpu())
    feature_mean, feature_std = matrix[train_rows].mean(0), matrix[train_rows].std(0).clamp_min(1e-6)
    matrix = ((matrix - feature_mean) / feature_std).to(device)

    heads, seed_rows, histories, standardised = [], [], {}, []
    for seed in cfg["seeds"]:
        head, history = train_head(cfg, matrix, train, dev, seed, device)
        with torch.no_grad():
            raw = head["model"](matrix)[0].float().cpu().numpy()
        score_mean, score_std = float(raw.mean()), float(raw.std() or 1.0)
        standardised.append((raw - score_mean) / score_std)
        seed_scores = dict(zip(ids, raw.tolist()))
        accuracy = {split: result["accuracy"] for split, result in
                    evaluate_scores(seed_scores, pairs, rankings, splits=SPLITS).items()}
        seed_rows.append({"seed": seed, "best_epoch": head["best_epoch"], "dev_loss": head["dev_loss"],
                          **{f"{split}_accuracy": accuracy[split] for split in SPLITS}})
        histories[seed] = history
        heads.append({"seed": seed, "state_dict": head["model"].state_dict(), "score_mean": score_mean,
                      "score_std": score_std})
    ensemble = np.mean(standardised, axis=0)
    ensemble_scores = dict(zip(ids, ensemble.tolist()))
    seed_summary = {f"{split}_{statistic}": round(float(function([row[f"{split}_accuracy"] for row in seed_rows])), 4)
                    for split in SPLITS for statistic, function in (("mean", np.mean), ("sd", np.std))}

    out = SCORER_DIR / "runs" / cfg["name"]
    metrics = {"name": cfg["name"], "config": cfg, "version": version, "device": device,
               "features": features.shape[1], "edits": len(ids), "seconds": round(time.monotonic() - started, 1),
               "seeds": seed_rows, "seed_summary": seed_summary,
               "ensemble": evaluate_scores(ensemble_scores, pairs, rankings, splits=SPLITS), "history": histories}
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "scores.jsonl", ({"image_id": image_id, "score": round(float(score), 5)}
                                       for image_id, score in zip(ids, ensemble)))
    torch.save({"config": cfg, "heads": heads, "feature_mean": feature_mean, "feature_std": feature_std},
               out / "head.pt")
    return {"name": cfg["name"], "features": features.shape[1], "edits": len(ids), "seconds": metrics["seconds"],
            "best_epochs": [row["best_epoch"] for row in seed_rows], **seed_summary,
            "ensemble_dev": metrics["ensemble"]["dev"]["accuracy"],
            "ensemble_test": metrics["ensemble"]["test"]["accuracy"]}
