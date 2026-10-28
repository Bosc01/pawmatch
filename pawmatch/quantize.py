"""Int8 copies of the two service models, measured against fp32 on the CPU.

Fixed before any held-out measurement (decisions.md). Nothing is chosen from these
numbers, and the service keeps its fp32 models.

- Detector (pawmatch.headdet): post-training static int8 for the MobileNetV3 body and
  the FPN (FX graph mode, PyTorch's default qnnpack settings), calibrated on the first
  CALIBRATION half-A training photos, and dynamic int8 for the box head's Linear
  layers. QNNPACK rejects the FPN's extra 1x1 stride-2 max pool, so that op stays fp32,
  as do the RPN head, RoIAlign, and NMS. The score threshold stays detect.THRESHOLD.
- Re-ID: the service embedding model ([serve] model and checkpoint: frozen DINOv2
  ViT-S/14 for DogFaceNet), dynamic int8 Linear layers with the default settings. The
  patch embedding, LayerNorm, and attention softmax stay fp32.

Quantized kernels run on the CPU only, so fp32 is measured there too, with the qnnpack
engine (fbgemm is x86 only) at torch's default thread count. Latency is batch 1,
forward pass only, on photos already decoded and transformed: LATENCY_PHOTOS photos
drawn with seed SEED, each timed once per run after WARMUP untimed photos, over
LATENCY_RUNS runs with fp32 and int8 interleaved. A run gives p50 and p95 over photos,
and the reported value is the median over runs.

Usage: python -m pawmatch.quantize CONFIG [detector] [reid] [report]   (no job: both runs;
report alone re-renders the markdown from the JSON files)
Writes results/quantization.json (detector, Oxford-IIIT Pet),
<results_dir>/quantization_reid.json (re-ID; results/dev for DogFaceNet), and
results/quantization.md from both.
"""
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.ao.quantization import get_default_qconfig_mapping, quantize_dynamic
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms.functional import to_tensor

from pawmatch import data, detect, embed, evaluate, headdet, metrics

ENGINE = "qnnpack"
CPU = torch.device("cpu")
CALIBRATION, LATENCY_PHOTOS, LATENCY_RUNS, WARMUP, SEED = 100, 100, 5, 10, 0
DETECTOR_OUT, MD_OUT = Path("results/quantization.json"), Path("results/quantization.md")
HALF_B_LOOK = ("third look at half B, after the OWL-ViT choice (detection_oxford.json) and the fp32 trained "
               "detector (detection_oxford_trained.json); nothing is chosen from it")
REID_LOOK = "DogFaceNet test split: development data, looked at before (m1 and later results); nothing is chosen from it"


def _open(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def _size(path):
    with Image.open(path) as im:
        return im.size


def _unfreeze(module):
    """FrozenBatchNorm2d to the equal eval-mode BatchNorm2d, which FX folds into the conv before it."""
    for name, child in module.named_children():
        if isinstance(child, FrozenBatchNorm2d):
            bn = torch.nn.BatchNorm2d(len(child.weight), eps=child.eps).eval()
            for k in ("weight", "bias", "running_mean", "running_var"):
                getattr(bn, k).data.copy_(getattr(child, k))
            setattr(module, name, bn)
        else:
            _unfreeze(child)


@torch.no_grad()
def int8_detector(model, calibration):
    """Static int8 body and FPN calibrated on PIL photos, then dynamic int8 Linear
    layers. Converts model in place and returns it."""
    torch.backends.quantized.engine = ENGINE
    model.to(CPU).eval()
    _unfreeze(model.backbone)
    x = lambda im: model.transform([to_tensor(im)])[0].tensors
    qconfig = get_default_qconfig_mapping(ENGINE).set_module_name("fpn.extra_blocks", None)  # 1x1 pool: QNNPACK refuses
    prepared = prepare_fx(model.backbone, qconfig, (x(calibration[0]),))
    for im in calibration:
        prepared(x(im))
    model.backbone = convert_fx(prepared)
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)


def int8_reid(model):
    """Dynamic int8 Linear layers on a copy of model."""
    torch.backends.quantized.engine = ENGINE
    return quantize_dynamic(model.to(CPU).eval(), {torch.nn.Linear}, dtype=torch.qint8)


def size_mb(model):
    """Serialized state_dict size in MB (10^6 bytes)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


@torch.no_grad()
def latency(models, inputs):
    """{name: median over runs of per-run p50 and p95 ms, and the runs}; models interleaved run by run."""
    runs = {name: [] for name in models}
    for _ in range(LATENCY_RUNS):
        for name, model in models.items():
            for x in inputs[:WARMUP]:
                model(x)
            ms = []
            for x in inputs:
                start = time.perf_counter()
                model(x)
                ms.append(1000 * (time.perf_counter() - start))
            runs[name].append({"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95))})
    return {name: {"p50_ms": float(np.median([r["p50_ms"] for r in rs])),
                   "p95_ms": float(np.median([r["p95_ms"] for r in rs])), "runs": rs} for name, rs in runs.items()}


def _changed(fp32, int8):
    """Items whose hit flips between the models: right only in fp32, right only in int8."""
    return {"fp32_only": int((fp32 & ~int8).sum()), "int8_only": int((int8 & ~fp32).sum())}


def _setup(commit):
    return {"code_commit": commit, "hardware": evaluate.hardware(), "torch": torch.__version__, "device": "cpu",
            "engine": ENGINE, "threads": torch.get_num_threads(),
            "latency_protocol": {"batch": 1, "photos": LATENCY_PHOTOS, "seed": SEED, "warmup": WARMUP,
                                 "runs": LATENCY_RUNS, "timed": "forward pass only, photo already decoded and transformed"}}


def calibration_photos():
    """Oxford half A only: the first CALIBRATION photos of the part the detector trained on."""
    dogs, train_idx, _ = headdet.half_a_split()
    return [dogs[i][0] for i in train_idx[:CALIBRATION]]


def detector():
    commit = evaluate.commit(["pawmatch"])  # before the run: later edits must not change the stamp
    dogs, train_idx, val_idx = headdet.half_a_split()
    _, _, halves = detect.oxford_halves()
    fp32 = headdet.HeadDetector(detect.THRESHOLD, device=CPU)
    int8 = headdet.HeadDetector(detect.THRESHOLD, device=CPU)
    int8.model = int8_detector(int8.model, [_open(p) for p in calibration_photos()])
    sample = random.Random(SEED).sample(val_idx, LATENCY_PHOTOS)
    lat = latency({"fp32": fp32.model, "int8": int8.model}, [[to_tensor(_open(dogs[i][0]))] for i in sample])
    old = json.loads(Path("results/detection_oxford_trained.json").read_text())
    committed = {"half_a_internal_val": next(r["recall_at_iou50"] for r in old["internal_val_threshold_grid"]
                                             if r["threshold"] == detect.THRESHOLD),
                 "half_b": old["half_b_result"]["recall_at_iou50"]["value"]}
    recall = {}
    for name, label, idx, look in (("half_a_internal_val", "half A internal validation", val_idx, "selection set, not one-shot"),
                                   ("half_b", "Oxford half B", halves["B"], HALF_B_LOOK)):
        paths, truths = [dogs[i][0] for i in idx], [dogs[i][1] for i in idx]
        sizes = [_size(p) for p in paths]
        r, hits = {"set": label, "look": look, "n": len(idx), "committed_fp32_mps": committed[name]}, {}
        for k, det in (("fp32", fp32), ("int8", int8)):
            h, raw, routes = detect.recall_with_rule(detect.best_boxes(det, paths, batch=headdet.EVAL_BATCH),
                                                     sizes, truths, detect.THRESHOLD)
            hits[k] = h
            r[k] = {"recall_at_iou50": {"value": float(h.mean()), "ci95": metrics.bootstrap_ci(
                        lambda w: np.average(h, weights=w), np.arange(len(h)))},
                    "recall_at_iou50_detector_box_only": float(raw.mean()), "routes": detect.shares(routes)}
            print(name, k, f"recall {h.mean():.4f}")
        r["photos_changed"] = _changed(hits["fp32"], hits["int8"])
        recall[name] = r
    return {"model": "trained dog-head detector (pawmatch.headdet): Faster R-CNN, MobileNetV3-Large FPN", **_setup(commit),
            "quantized": "static int8 MobileNetV3 body and FPN (FX graph mode, default qnnpack settings); dynamic int8 box head Linear layers",
            "fp32_kept": "FPN extra 1x1 stride-2 max pool (QNNPACK rejects 1x1 pooling), RPN head, RoIAlign, NMS, box decoding",
            "calibration": f"first {CALIBRATION} half A training photos in headdet.half_a_split order",
            "threshold": detect.THRESHOLD, "size_mb": {"fp32": size_mb(fp32.model), "int8": size_mb(int8.model)},
            "latency": {"photos": f"half A internal validation, {LATENCY_PHOTOS} drawn with seed {SEED}", **lat},
            "recall": recall}


def reid(cfg):
    commit = evaluate.commit(["pawmatch"])
    splits, sha = data.load_splits(cfg)
    s, p = cfg["serve"], cfg["protocol"]
    fp32, tf = embed.build(s["model"], s.get("checkpoint") or None)
    fp32.to(CPU).eval()
    int8 = int8_reid(fp32)
    sample = random.Random(SEED).sample([path for path, _ in splits["val_a"]], LATENCY_PHOTOS)
    lat = latency({"fp32": fp32, "int8": int8}, [tf(_open(path))[None] for path in sample])
    items = splits["test"]
    pairs = data.official_pairs(cfg, items) or data.verification_pairs(items, p["seed"])
    emb, test, hits = {}, {}, {}
    for k, model in (("fp32", fp32), ("int8", int8)):
        emb[k] = embed.encode(model, tf, [path for path, _ in items], dev=CPU)
        test[k], per_query = evaluate.compute(items, emb[k], pairs, p["seed"], p["unknown_frac"], p["bootstrap"])
        hits[k] = per_query["rank1"]
        print(k, f"rank-1 {test[k]['closed_set']['rank1']['value']:.4f}")
    cos = (emb["fp32"] * emb["int8"]).sum(1)
    m1 = cfg["results_dir"] / f"m1_{s['model']}.json"
    committed = json.loads(m1.read_text())["closed_set"]["rank1"]["value"] if m1.exists() and not s.get("checkpoint") else None
    return {"dataset": cfg["dataset"]["name"], "model": s["model"], "checkpoint": s.get("checkpoint", ""),
            **_setup(commit), "quantized": "dynamic int8 Linear layers (default settings)",
            "fp32_kept": "patch embedding conv, LayerNorm, attention softmax, GELU",
            "size_mb": {"fp32": size_mb(fp32), "int8": size_mb(int8)},
            "latency": {"photos": f"val_a, {LATENCY_PHOTOS} drawn with seed {SEED}", **lat},
            "split": "test", "split_sha256": sha, "look": REID_LOOK, "protocol": "Milestone 1 (pawmatch.evaluate)",
            "fp32": test["fp32"], "int8": test["int8"], "committed_fp32_mps_rank1": committed,
            "rank1_queries_changed": _changed(hits["fp32"], hits["int8"]),
            "cosine_fp32_int8": {"mean": float(cos.mean()), "min": float(cos.min())}}


def _pct(m):
    return f"{100 * m['value']:.2f}% [{100 * m['ci95'][0]:.2f}, {100 * m['ci95'][1]:.2f}]"


def _model_section(r, rows):
    """Common table: size, latency, then the model's accuracy rows (label, fp32 metric, int8 metric)."""
    lat = r["latency"]
    out = ["| | fp32 | int8 | int8 against fp32 |", "|---|---|---|---|",
           f"| Size (MB) | {r['size_mb']['fp32']:.2f} | {r['size_mb']['int8']:.2f} | "
           f"{r['size_mb']['int8'] / r['size_mb']['fp32']:.2f}x |"]
    for q in ("p50", "p95"):
        a, b = lat["fp32"][f"{q}_ms"], lat["int8"][f"{q}_ms"]
        out.append(f"| {q} latency (ms) | {a:.1f} | {b:.1f} | {b / a:.2f}x |")
    for label, a, b in rows:
        out.append(f"| {label} | {_pct(a)} | {_pct(b)} | {100 * (b['value'] - a['value']):+.2f} points |")
    out.append("")
    for q in ("p50", "p95"):
        a, b = ([run[f"{q}_ms"] for run in lat[k]["runs"]] for k in ("fp32", "int8"))
        faster = sum(y < x for x, y in zip(a, b))
        out.append(f"- {q} per run (ms): fp32 {', '.join(f'{v:.1f}' for v in a)}; int8 {', '.join(f'{v:.1f}' for v in b)}. "
                   f"int8 was faster in {faster} of {len(a)} runs.")
    out.append(f"- Latency photos: {lat['photos']}.")
    return out


def render(det, rid):
    out = ["# Int8 quantization of the service models", "",
           "Generated by `pawmatch/quantize.py` from `results/quantization.json` (detector) and "
           "`results/dev/quantization_reid.json` (re-ID). Do not edit by hand. Commands, from the repository root:", "",
           "    ./venv/bin/python -m pawmatch.quantize configs/dogfacenet.toml detector",
           "    ./venv/bin/python -m pawmatch.quantize configs/dogfacenet.toml reid",
           "    ./venv/bin/python -m pawmatch.quantize configs/dogfacenet.toml report   # re-renders this file only", "",
           "The quantization settings were fixed before any held-out measurement, and nothing was chosen from "
           "these numbers. The service still runs the fp32 models. Accuracy cells show the value and its 95% "
           "bootstrap interval.", ""]
    for r in (det, rid):
        if r:
            lp = r["latency_protocol"]
            out += ["## Setup", "",
                    f"- {r['hardware']}, PyTorch {r['torch']}, CPU only, {r['engine']} engine, {r['threads']} threads. "
                    "Quantized kernels have no MPS version, so fp32 is measured on the CPU too. torch.ao.quantization "
                    "is deprecated in this PyTorch (torchao is the migration path), though it still runs.",
                    "- Size: serialized state_dict, MB = 10^6 bytes.",
                    f"- Latency: batch {lp['batch']}, {lp['timed']}. {lp['photos']} photos drawn with seed {lp['seed']}. "
                    f"Each run times every photo once after {lp['warmup']} untimed photos, over {lp['runs']} runs with "
                    "fp32 and int8 interleaved. p50 and p95 are over photos within a run; the table shows the median "
                    "over runs.", ""]
            break
    if det:
        rec = det["recall"]
        out += ["## Detector", "", f"{det['model']}. Code commit {det['code_commit']}.", "",
                f"- int8: {det['quantized']}.", f"- Kept in fp32: {det['fp32_kept']}.",
                f"- Calibration: {det['calibration']}. These are Oxford half A photos only, from the part the detector "
                "trained on: no half B photo and no internal-validation photo (tests/test_quantize.py checks this).",
                f"- Score threshold: {det['threshold']}, the service threshold, not re-selected for int8.", ""]
        out += _model_section(det, [(f"Recall at IoU 0.5, {v['set']} (n={v['n']})", v["fp32"]["recall_at_iou50"],
                                     v["int8"]["recall_at_iou50"]) for v in rec.values()])
        for v in rec.values():
            ch = v["photos_changed"]
            out.append(f"- {v['set']} ({v['look']}): {ch['fp32_only']} photos hit only in fp32, "
                       f"{ch['int8_only']} only in int8. Committed fp32 recall on MPS: {100 * v['committed_fp32_mps']:.2f}%. "
                       "Detector box only, fp32 then int8: "
                       f"{100 * v['fp32']['recall_at_iou50_detector_box_only']:.2f}%, "
                       f"{100 * v['int8']['recall_at_iou50_detector_box_only']:.2f}%.")
            for m in ("fp32", "int8"):
                out.append(f"  - {m} routes: " + ", ".join(f"{route} {100 * share:.2f}%"
                                                           for route, share in v[m]["routes"].items()) + ".")
        out.append("")
    if rid:
        f, i = rid["fp32"], rid["int8"]
        c, o = f["closed_set"], f["open_set"]
        rows = [("Rank-1, closed set (DogFaceNet development)", ("closed_set", "rank1")),
                ("Rank-5, closed set", ("closed_set", "rank5")),
                ("MRR, closed set", ("closed_set", "mrr")), ("mAP, leave one out", ("map_leave_one_out",)),
                ("Verification AUC", ("verification", "auc")),
                ("Verification AUC, clamped as the paper", ("verification", "auc_clamped_as_paper")),
                ("Rank-1, open set, known dogs", ("open_set", "rank1_known")),
                ("Detection AUROC, open set", ("open_set", "detection_auroc"))]
        get = lambda r, keys: r[keys[0]] if len(keys) == 1 else r[keys[0]][keys[1]]
        ch, cos = rid["rank1_queries_changed"], rid["cosine_fp32_int8"]
        out += ["## Re-ID embedding model: DogFaceNet development numbers", "",
                f"{rid['model']}, checkpoint {rid['checkpoint'] or 'none (frozen backbone)'}: the model `[serve]` uses. "
                f"Code commit {rid['code_commit']}.", "",
                "- Every number in this section, rank-1 before and after included, is a DogFaceNet development number. "
                "It lives in `results/dev/quantization_reid.json`, which `pawmatch.report` never reads, so it is never "
                "offered as a resume fact (tests/test_report.py checks this).",
                f"- int8: {rid['quantized']}.", f"- Kept in fp32: {rid['fp32_kept']}.",
                f"- Evaluation: {rid['protocol']}, whole photo resized to 224 px with no detection. {rid['look']}.", ""]
        out += _model_section(rid, [(label, get(f, keys), get(i, keys)) for label, keys in rows])
        out += [f"- Counts: closed set {c['n_queries']} queries of {c['n_query_dogs']} dogs against {c['n_gallery']} "
                f"gallery photos; open set {o['n_known_queries']} known and {o['n_unknown_queries']} unknown queries "
                f"against {o['n_gallery']}; mAP over {f['map_leave_one_out']['n_queries']} photos; "
                f"{f['verification']['n_pairs']} verification pairs.",
                f"- Closed-set rank-1 changed for {ch['fp32_only'] + ch['int8_only']} queries: {ch['fp32_only']} right "
                f"only in fp32, {ch['int8_only']} right only in int8.",
                f"- Cosine similarity between the fp32 and int8 embedding of each test photo: mean {cos['mean']:.4f}, "
                f"min {cos['min']:.4f}."]
        if rid["committed_fp32_mps_rank1"] is not None:
            out.append(f"- Committed fp32 rank-1 on MPS (m1_{rid['model']}.json): "
                       f"{100 * rid['committed_fp32_mps_rank1']:.2f}%.")
        out.append("")
    return "\n".join(out)


def main(config, *jobs):
    cfg = data.load_config(config)
    runs = {"detector": detector, "reid": lambda: reid(cfg)}
    outs = {"detector": DETECTOR_OUT, "reid": cfg["results_dir"] / "quantization_reid.json"}
    for job in jobs or runs:
        if job != "report":
            outs[job].write_text(json.dumps(runs[job](), indent=1) + "\n")
            print(outs[job])
    load = lambda path: json.loads(path.read_text()) if path.exists() else None
    MD_OUT.write_text(render(load(outs["detector"]), load(outs["reid"])))
    print(MD_OUT)


if __name__ == "__main__":
    main(*sys.argv[1:])
