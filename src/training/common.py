"""Shared fine-tuning loop for sequence classifiers (sentiment and events).

A plain PyTorch loop (AdamW, linear warm-up/decay, gradient clipping, best-epoch
checkpointing on validation macro-F1). Runs on CUDA, Apple MPS or CPU, and the
same code backs the Colab notebooks in ``src/training/notebooks``.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

log = logging.getLogger("train")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainConfig:
    base_model: str
    out_name: str
    labels: list[str]
    epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 32
    max_len: int = 128
    warmup: float = 0.06
    weight_decay: float = 0.01
    class_weighted: bool = False
    freeze_embeddings: bool = False  # XLM-R: 192M of its 278M params are the 250k-token embedding table
    seed: int = 42


def encode(tok, entities: list[str], texts: list[str], max_len: int):  # type: ignore[no-untyped-def]
    """Entity-aware input: ``<entity> [SEP] <sentence>``; entity "" for document-level rows."""
    # Pad to multiples of 32 so Apple MPS caches a handful of graph shapes, not one per length.
    return tok(
        entities,
        texts,
        truncation="only_second",
        max_length=max_len,
        padding=True,
        pad_to_multiple_of=32,
        return_tensors="pt",
    )


@torch.inference_mode()
def predict_logits(
    model, tok, entities: list[str], texts: list[str], max_len: int, batch_size: int = 64
) -> np.ndarray:  # type: ignore[no-untyped-def]
    dev = next(model.parameters()).device
    model.eval()
    out = []
    order = np.argsort([len(t) for t in texts])
    for b in range(0, len(texts), batch_size):
        idx = order[b : b + batch_size]
        enc = encode(tok, [entities[i] for i in idx], [texts[i] for i in idx], max_len)
        logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits.float().cpu().numpy()
        out.append((idx, logits))
    res = np.zeros((len(texts), out[0][1].shape[1]), dtype=np.float32)
    for idx, logits in out:
        res[idx] = logits
    return res


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al. 2017): one scalar T minimising validation NLL."""
    lt = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():  # type: ignore[no-untyped-def]
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lt / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def train_classifier(cfg: TrainConfig, train: pd.DataFrame, val: pd.DataFrame) -> Path:
    """``train``/``val`` need columns ``text``, ``entity`` and integer ``y`` (index into cfg.labels)."""
    from engine.diskguard import require_space

    require_space(f"train {cfg.out_name}", 1.2)  # checkpoint (overwritten in place)
    seed_everything(cfg.seed)
    dev = device()
    out_dir = MODELS_DIR / cfg.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    id2label = dict(enumerate(cfg.labels))
    base_cfg = AutoConfig.from_pretrained(cfg.base_model)
    base_names = [str(base_cfg.id2label[i]).lower() for i in range(len(base_cfg.id2label))]
    if sorted(base_names) == sorted(cfg.labels):
        # Reuse a pretrained head whose labels match ours by name (FinBERT is pos/neg/neu):
        # permute its rows into our label order instead of re-initialising it.
        model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model)
        head = model.classifier
        order = [base_names.index(lbl) for lbl in cfg.labels]
        with torch.no_grad():
            head.weight.copy_(head.weight[order].clone())
            head.bias.copy_(head.bias[order].clone())
        model.config.id2label = id2label
        model.config.label2id = {v: k for k, v in id2label.items()}
        log.info("reused pretrained head, permuted %s -> %s", base_names, cfg.labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model,
            num_labels=len(cfg.labels),
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
            ignore_mismatched_sizes=True,
        )
    if cfg.freeze_embeddings:
        emb = model.get_input_embeddings()
        emb.weight.requires_grad_(False)
        log.info("froze input embeddings (%.0fM params)", emb.weight.numel() / 1e6)
    model.to(dev)

    weights = None
    if cfg.class_weighted:
        counts = np.bincount(train["y"], minlength=len(cfg.labels)).astype(float)
        w = 1.0 / np.sqrt(np.maximum(counts, 1))
        weights = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm")
    params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(k in n for k in no_decay)
            ],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup * total), total)

    best_f1, history = -1.0, []
    rng = np.random.default_rng(cfg.seed)
    texts, ents, ys = train["text"].tolist(), train["entity"].tolist(), train["y"].to_numpy()
    for epoch in range(cfg.epochs):
        model.train()
        perm = rng.permutation(len(train))
        t0, running = time.time(), 0.0
        for step in range(steps_per_epoch):
            idx = perm[step * cfg.batch_size : (step + 1) * cfg.batch_size]
            enc = encode(tok, [ents[i] for i in idx], [texts[i] for i in idx], cfg.max_len)
            logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits
            loss = loss_fn(logits, torch.tensor(ys[idx], device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += loss.item()
            if dev.type == "mps" and step % 100 == 0:
                torch.mps.empty_cache()
            if step % 200 == 0:
                log.info(
                    "epoch %d step %d/%d loss %.4f (%.0fs)",
                    epoch + 1,
                    step,
                    steps_per_epoch,
                    running / (step + 1),
                    time.time() - t0,
                )
        vl = predict_logits(model, tok, val["entity"].tolist(), val["text"].tolist(), cfg.max_len)
        f1 = f1_score(val["y"], vl.argmax(1), average="macro")
        acc = float((vl.argmax(1) == val["y"].to_numpy()).mean())
        history.append(
            {
                "epoch": epoch + 1,
                "val_macro_f1": round(float(f1), 4),
                "val_acc": round(acc, 4),
                "train_loss": round(running / steps_per_epoch, 4),
                "seconds": round(time.time() - t0),
            }
        )
        log.info("epoch %d: val macro-F1 %.4f acc %.4f", epoch + 1, f1, acc)
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)

    best = AutoModelForSequenceClassification.from_pretrained(out_dir).to(dev)
    vl = predict_logits(best, tok, val["entity"].tolist(), val["text"].tolist(), cfg.max_len)
    temperature = fit_temperature(vl, val["y"].to_numpy())
    meta = {
        "config": asdict(cfg),
        "history": history,
        "best_val_macro_f1": round(best_f1, 4),
        "temperature": temperature,
        "n_train": len(train),
        "n_val": len(val),
        "device": str(dev),
    }
    (out_dir / "training.json").write_text(json.dumps(meta, indent=2))
    log.info("saved %s (best val macro-F1 %.4f, T=%.3f)", out_dir, best_f1, temperature)
    return out_dir
