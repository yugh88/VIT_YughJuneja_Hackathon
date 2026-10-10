"""ONNX Runtime sequence classifier - the only inference dependency the engine needs.

Loads ``models/onnx/<name>/{model.int8.onnx, tokenizer.json, sentinel.json}`` as
written by ``training.export_onnx``. Inputs are (entity, text) pairs encoded as
``<entity> [SEP] <text>``; an empty entity gives document-level scoring.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from engine.config import get_settings

log = logging.getLogger(__name__)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class OnnxClassifier:
    def __init__(self, model_dir: Path, threads: int | None = None) -> None:
        self.dir = model_dir
        self.cfg: dict[str, Any] = json.loads((model_dir / "sentinel.json").read_text(encoding="utf-8"))
        self.labels: list[str] = self.cfg["labels"]
        self.max_len: int = int(self.cfg.get("max_len", 128))
        self.temperature: float = float(self.cfg.get("temperature", 1.0))
        self.version: str = self.cfg.get("version", model_dir.name)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or get_settings().onnx_threads or min(4, os.cpu_count() or 4)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # CPU is the default everywhere; SENTINEL_ONNX_PROVIDER=coreml tries Apple's CoreML first (D75).
        providers = ["CPUExecutionProvider"]
        if os.environ.get("SENTINEL_ONNX_PROVIDER", "").lower() == "coreml":
            providers = ["CoreMLExecutionProvider", *providers]
        self.session = ort.InferenceSession(str(model_dir / "model.int8.onnx"), opts, providers=providers)
        self._cache: OrderedDict[bytes, np.ndarray] = OrderedDict()  # (entity, text) -> logits, LRU
        self.cache_hits = 0
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # NLI models take (premise, hypothesis) with a long premise first: truncate whichever is longer.
        nli = "entailment" in [lbl.lower() for lbl in self.labels]
        self.tok.enable_truncation(self.max_len, strategy="longest_first" if nli else "only_second")
        pad_id = self.tok.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = self.tok.token_to_id("<pad>") or 0
        self.tok.enable_padding(pad_id=pad_id)

    CACHE_SIZE = 50_000

    def logits(self, entities: list[str], texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        keys = [
            hashlib.blake2b(f"{e or ''}\x1f{t}".encode(), digest_size=16).digest()
            for e, t in zip(entities, texts, strict=True)
        ]
        todo = []
        for i, k in enumerate(keys):  # re-polled feeds and syndicated copies repeat the same text (D75)
            hit = self._cache.get(k)
            if hit is None:
                todo.append(i)
            else:
                out[i] = hit
                self._cache.move_to_end(k)
                self.cache_hits += 1
        todo_arr = np.array(todo, dtype=np.int64)
        order = (
            todo_arr[np.argsort([len(texts[i]) for i in todo])] if todo else todo_arr
        )  # length-bucketed batches pad less
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            encs = self.tok.encode_batch([(entities[i] or "", texts[i]) for i in idx])
            feed = {
                "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
            }
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)
            out[idx] = self.session.run(["logits"], feed)[0]
            for i in idx:
                self._cache[keys[i]] = out[i].copy()
        while len(self._cache) > self.CACHE_SIZE:
            self._cache.popitem(last=False)
        return out

    def predict_proba(self, entities: list[str], texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        return softmax(self.logits(entities, texts) / self.temperature)

    def calibrate(
        self, raw: np.ndarray, intensity: np.ndarray | None = None, which: str | None = None
    ) -> np.ndarray:
        """Map raw P(pos)-P(neg) (and text intensity) to the published score.

        ``which`` = "range" (full -1..1 range; default) or "human" (FiQA-faithful linear
        scale); see training.calibrate_sentiment for how each was chosen.
        """
        cal = self.cfg.get("calibration") or {}
        if "published" in cal:
            cal = cal[which or cal["published"]]
        scale = float(cal.get("scale", 1.0))
        if cal.get("type") == "intensity":
            inten = np.zeros_like(raw) if intensity is None else intensity
            mag = np.abs(raw) ** float(cal["power"]) * (scale + float(cal["beta"]) * inten)
            out = np.sign(raw) * np.minimum(1.0, mag)
        else:
            out = raw * scale
        return np.clip(out, -1.0, 1.0)
