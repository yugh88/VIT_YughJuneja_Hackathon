"""Export fine-tuned classifiers to ONNX and quantise them to INT8.

    python -m training.export_onnx sentiment-en sentiment-multi event-multi
    python -m training.export_onnx --zero-shot     # multilingual NLI fallback for events

Each export lands in ``models/onnx/<name>/`` with ``model.int8.onnx``,
``tokenizer.json`` and ``sentinel.json`` (labels, max length, temperature and,
for sentiment, the score calibration). The engine needs only onnxruntime +
tokenizers at runtime - no PyTorch.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.common import MODELS_DIR

log = logging.getLogger("export")
ZERO_SHOT = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


class _Wrap(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, with_tt: bool) -> None:
        super().__init__()
        self.model = model
        self.with_tt = with_tt

    def forward(self, input_ids, attention_mask, token_type_ids=None):  # type: ignore[no-untyped-def]
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.with_tt:
            kw["token_type_ids"] = token_type_ids
        return self.model(**kw).logits


def _run(path: Path, feed: dict[str, np.ndarray]) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(["logits"], feed)[0]  # type: ignore[no-any-return]


def _probs(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)  # type: ignore[no-any-return]


def quantise_with_fidelity(fp32: Path, out: Path, feed: dict[str, np.ndarray]) -> dict:  # type: ignore[type-arg]
    """Full dynamic INT8 when it preserves predictions; otherwise quantise only the
    embedding table (Gather). DeBERTa-v3 loses most of its contrast under full INT8
    (measured: entailment logit 1.61 -> 0.45), so the choice is made per model, on data."""
    ref = _probs(_run(fp32, feed))
    report = {}
    for mode, kw in (("int8-full", {}), ("int8-embeddings", {"op_types_to_quantize": ["Gather"]})):
        quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8, **kw)
        q = _probs(_run(out, feed))
        agree = float((q.argmax(1) == ref.argmax(1)).mean())
        mad = float(np.abs(q - ref).mean())
        report[mode] = {"argmax_agreement": round(agree, 4), "mean_abs_prob_diff": round(mad, 4)}
        log.info("%s: agreement %.4f, mean |dp| %.4f", mode, agree, mad)
        if agree >= 0.98 and mad <= 0.02:
            return {"quantization": mode, "fidelity_vs_fp32": report, "fidelity_rows": len(ref)}
    return {"quantization": "int8-embeddings", "fidelity_vs_fp32": report, "fidelity_rows": len(ref)}


def fp16_storage(path: Path, min_elems: int = 65_536) -> int:
    """Store large float32 weight matrices as float16 behind a Cast node.

    ONNX Runtime constant-folds Cast(initializer) at session load, so compute stays
    float32 while the file shrinks ~35%. Measured on sentiment-multi: 535 -> 364 MB,
    argmax agreement 1.0, mean |dp| 7e-5 on 400 validation rows.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    casts = []
    for init in list(graph.initializer):
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= min_elems:
            half = numpy_helper.from_array(
                numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16"
            )
            graph.initializer.remove(init)
            graph.initializer.append(half)
            casts.append(
                helper.make_node(
                    "Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"
                )
            )
    for c in reversed(casts):
        graph.node.insert(0, c)
    onnx.save(model, str(path))
    return len(casts)


def export(src: Path | str, out: Path, max_len: int, extra: dict, pairs: list[tuple[str, str]]) -> Path:  # type: ignore[type-arg]
    """``pairs`` are (first, second) segments from validation data, used for the fidelity check."""
    from engine.diskguard import require_space

    require_space(f"export {out.name}", 1.6)  # temporary FP32 graph + quantised copies
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src, dtype=torch.float32).eval()
    enc = tok(["Acme"], ["Acme shares fell after a profit warning."], return_tensors="pt")
    with_tt = "token_type_ids" in enc and model.config.model_type not in {
        "xlm-roberta",
        "roberta",
        "deberta-v2",
    }
    names = ["input_ids", "attention_mask"] + (["token_type_ids"] if with_tt else [])
    args = tuple(enc[n] for n in names)
    fp32 = out / "model.onnx"
    torch.onnx.export(
        _Wrap(model, with_tt),
        args,
        str(fp32),
        input_names=names,
        output_names=["logits"],
        dynamic_axes={n: {0: "batch", 1: "seq"} for n in names} | {"logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    fe = tok(
        [a for a, _ in pairs],
        [b for _, b in pairs],
        truncation="only_second",
        max_length=max_len,
        padding=True,
        return_tensors="np",
    )
    feed = {n: fe[n].astype(np.int64) for n in names}
    q = quantise_with_fidelity(fp32, out / "model.int8.onnx", feed)
    before = _probs(_run(out / "model.int8.onnx", feed))
    q["fp16_weight_tensors"] = fp16_storage(out / "model.int8.onnx")
    after = _probs(_run(out / "model.int8.onnx", feed))
    q["fp16_storage_agreement"] = round(float((before.argmax(1) == after.argmax(1)).mean()), 4)
    fp32.unlink()
    for f in out.glob("*.data"):
        f.unlink()
    tok.save_pretrained(out / "_tok")
    shutil.copy(out / "_tok" / "tokenizer.json", out / "tokenizer.json")
    shutil.rmtree(out / "_tok")
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    cfg = {
        "labels": labels,
        "max_len": max_len,
        "inputs": names,
        "source": str(src),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **q,
        **extra,
    }
    (out / "sentinel.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    size = (out / "model.int8.onnx").stat().st_size / 1e6
    log.info("exported %s -> %s (%.0f MB, %s)", src, out, size, q["quantization"])
    return out


def fidelity_pairs(name: str, n: int = 256) -> list[tuple[str, str]]:
    from training.data import load_phrasebank, load_twitter_sentiment, load_twitter_topic

    if name.startswith("sentiment"):
        df = pd.concat([load_phrasebank(), load_twitter_sentiment()])
    else:
        df = load_twitter_topic().assign(entity="")
    df = df[df["split"] == "val"].sample(n, random_state=0)
    return [("", t) for t in df["text"]]


def nli_pairs(n: int = 24) -> list[tuple[str, str]]:
    from engine.nlp.events import CLASSES, DESCRIPTIONS
    from engine.schema import EventClass
    from training.data import load_twitter_topic

    texts = load_twitter_topic().query("split == 'val'").sample(n, random_state=0)["text"].tolist()
    return [(t, f"This text is about {DESCRIPTIONS[EventClass(c)]}.") for t in texts for c in CLASSES]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="fine-tuned model dirs under models/")
    ap.add_argument("--zero-shot", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for name in args.names:
        src = MODELS_DIR / name
        meta = json.loads((src / "training.json").read_text())
        max_len = meta["config"]["max_len"]
        extra = {
            "temperature": meta["temperature"],
            "best_val_macro_f1": meta["best_val_macro_f1"],
            "version": f"{name}-1.0",
        }
        export(src, MODELS_DIR / "onnx" / name, max_len, extra, fidelity_pairs(name))
        if name.startswith("sentiment"):
            from training.calibrate_sentiment import calibrate

            calibrate(MODELS_DIR / "onnx" / name)
    if args.zero_shot:
        export(
            ZERO_SHOT,
            MODELS_DIR / "onnx" / "zeroshot-nli",
            128,
            {"temperature": 1.0, "version": "mdeberta-xnli-zs-1.0", "hypothesis": "This text is about {}."},
            nli_pairs(),
        )


if __name__ == "__main__":
    main()
