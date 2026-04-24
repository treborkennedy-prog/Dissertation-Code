#Adapted from https://github.com/NYUSHCS/MolecularGPT

import re
import json
import csv
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from peft import PeftModel
from rdkit import Chem, DataStructs

#use available RDKit fingerprint generator
try:
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    HAS_MORGAN_GEN = True
except Exception:
    HAS_MORGAN_GEN = False
    from rdkit.Chem import AllChem

#Syntax patterns used to extract the query molecule and the numeric prediction from text
SMILES_PATTERN = re.compile(r"SMILES\s*:\s*([^\s]+)")
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
METRICS = ("MAE", "RMSE", "R2", "R2_test_trainmean")

def parse_grid(x, cast):
    xs = [cast(v) for v in x] if isinstance(x, (list, tuple)) else [cast(v.strip()) for v in str(x).split(",") if v.strip()]
    seen, out = set(), []
    for v in xs:
        k = f"{v:.12g}" if isinstance(v, float) else v
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out

#Fix RNGs so validation sweeps and generation are reproducible
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#take the final number from the model output
def extract_number(text):
    xs = NUMBER_PATTERN.findall(str(text))
    return float(xs[-1]) if xs else float("nan")

def extract_smiles(text):
    m = SMILES_PATTERN.search(str(text))
    return m.group(1).strip() if m else ""

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({
                "id": str(r["id"]),
                "instruction": str(r["instruction"]),
                "input": str(r["input"]),
                "output": float(r["output"]),
                "smiles": extract_smiles(r["input"]),
            })
    return rows

#fingerprint generation
def smiles_to_fp(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if HAS_MORGAN_GEN:
        return GetMorganGenerator(radius=radius, fpSize=nbits).GetFingerprint(mol)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

def build_fp_cache(records, radius=2, nbits=2048):
    return [smiles_to_fp(r["smiles"], radius=radius, nbits=nbits) for r in records]

#top-k retrieval by Tanimoto similarity
def topk_neighbors(query_fp, query_smiles, train_records, train_fps, k):
    if k <= 0:
        return []
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, train_fps)
    pairs = [(i, s) for i, s in enumerate(sims) if train_records[i]["smiles"] != query_smiles]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in pairs[:k]]

def build_prompt(instruction, examples, query_input):
    parts = [instruction.strip(), ""]
    for ex_input, ex_output in examples:
        parts += [ex_input.strip(), f"Answer: {ex_output}", ""]
    parts += [query_input.strip(), "Answer:"]
    return "\n".join(parts)

#ensure prompt is not too long, removing examples if needed
def fit_prompt(tokenizer, instruction, examples, query_input, context_limit, reserved_for_generation):
    max_tokens = context_limit - reserved_for_generation
    xs = list(examples)
    while True:
        prompt = build_prompt(instruction, xs, query_input)
        if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) <= max_tokens or not xs:
            return prompt, len(xs)
        xs.pop()

def compute_metrics(y_true, y_pred, baseline_mean=None):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[keep], yp[keep]
    err = yt - yp
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if baseline_mean is None:
        baseline_mean = float(np.mean(yt))
    ss_tot_train = float(np.sum((yt - float(baseline_mean)) ** 2))
    return {
        "n": int(len(yt)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot),
        "R2_test_trainmean": float("nan") if ss_tot_train == 0.0 else float(1.0 - ss_res / ss_tot_train),
    }

def predict_rows(model, tokenizer, records_to_predict, retrieval_records, retrieval_fps, k, temperature,
                 max_new_tokens, fp_radius, fp_nbits, context_limit, reserved_for_generation):
    gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens) if float(temperature) <= 0 else GenerationConfig(
        do_sample=True, temperature=float(temperature), max_new_tokens=max_new_tokens
    )
    rows = []
    for row in records_to_predict:
        q_fp = smiles_to_fp(row["smiles"], radius=fp_radius, nbits=fp_nbits)
        idxs = topk_neighbors(q_fp, row["smiles"], retrieval_records, retrieval_fps, k)
        examples = [(retrieval_records[i]["input"], retrieval_records[i]["output"]) for i in idxs]
        prompt, k_used = fit_prompt(tokenizer, row["instruction"], examples, row["input"], context_limit, reserved_for_generation)
        enc = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            out = model.generate(**enc, generation_config=gen_cfg)
        pred = extract_number(tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True))
        rows.append({
            **row,
            "prediction": pred,
            "k_requested": int(k),
            "k_used": int(k_used),
            "temperature": float(temperature),
        })
    return rows

def load_model(base_model, lora_weights, load_8bit):
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        load_in_8bit=load_8bit,
        device_map={"": 0},
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, str(lora_weights), torch_dtype=torch.float16, device_map={"": 0})
    model.eval()
    return model

#select best hyperparameter config
def choose_best(rows, metric):
    metric = metric.upper()

    def key(r):
        if metric == "MAE":
            return (r["val_MAE"], r["val_RMSE"], -r["val_R2"], r["step"], r["k"], r["temperature"])
        if metric == "R2":
            return (-r["val_R2"], r["val_RMSE"], r["val_MAE"], r["step"], r["k"], r["temperature"])
        return (r["val_RMSE"], r["val_MAE"], -r["val_R2"], r["step"], r["k"], r["temperature"])

    return min(rows, key=key)

def plot_metric(rows, key, ylabel, title, outpath):
    plt.figure(figsize=(9, 6))
    for step in sorted({r["step"] for r in rows}):
        sub = sorted((r for r in rows if r["step"] == step), key=lambda r: r["k"])
        plt.plot([r["k"] for r in sub], [r[key] for r in sub], marker="o", linewidth=2, label=f"checkpoint-{step}")
    plt.xlabel("k", fontsize=16, fontweight="bold")
    plt.ylabel(ylabel, fontsize=16, fontweight="bold")
    plt.title(title, fontsize=18, fontweight="bold")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.tick_params(axis="both", labelsize=13)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def read_eval_history(root):
    p = Path(root) / "trainer_state.json"
    if not p.exists():
        return []
    h = json.loads(p.read_text(encoding="utf-8")).get("log_history", [])
    out = []
    for r in h:
        if "eval_loss" in r and "step" in r:
            out.append({"step": int(r["step"]), "eval_loss": float(r["eval_loss"])})
    seen, uniq = set(), []
    for r in out:
        if r["step"] not in seen:
            uniq.append(r)
            seen.add(r["step"])
    return uniq

#shortlist checkpoints near the best checkpoint, plus outside checkpoints
def shortlist_steps(root, radius, outside):
    ckpts = sorted([p for p in Path(root).glob("checkpoint-*") if p.is_dir()], key=lambda p: int(p.name.split("-")[-1]))
    steps = [int(p.name.split("-")[-1]) for p in ckpts]
    hist = [r for r in read_eval_history(root) if r["step"] in set(steps)]
    if not steps:
        return [], [], ckpts
    if not hist:
        return steps, [], ckpts
    ranked = sorted(hist, key=lambda r: (r["eval_loss"], r["step"]))
    best = ranked[0]["step"]
    i = steps.index(best)
    core = steps[max(0, i - radius): min(len(steps), i + radius + 1)]
    rest = [s for s in steps if s not in core]
    extra = []
    if rest:
        extra.append(rest[0])
        if len(rest) > 1:
            extra.append(rest[-1])
        if len(rest) > 2:
            extra.append(rest[len(rest) // 2])
    extra = extra[:outside]
    return sorted(set(core + extra)), hist, ckpts

#evaluate selected checkpoints across k/temperature settings
def sweep_steps(args, steps, rows, train_records, train_fps, k_grid, temp_grid):
    if not steps:
        return []
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    out = []
    for step in steps:
        ckpt = Path(args.lora_root) / f"checkpoint-{step}"
        model = load_model(args.base_model, ckpt, args.load_8bit)
        for temperature in temp_grid:
            for k in k_grid:
                set_seed(args.seed + 100000 * step + 1000 * int(round(temperature * 1000)) + int(k))
                pred_rows = predict_rows(
                    model, tokenizer, rows, train_records, train_fps, k, temperature,
                    args.max_new_tokens, args.fp_radius, args.fp_nbits,
                    args.context_limit, args.reserved_for_generation,
                )
                m = compute_metrics(
                    [r["output"] for r in pred_rows],
                    [r["prediction"] for r in pred_rows],
                    baseline_mean=args.train_mean,
                )
                out.append({
                    "checkpoint": ckpt.name,
                    "path": str(ckpt),
                    "step": step,
                    "k": int(k),
                    "temperature": float(temperature),
                    **{f"val_{x}": m[x] for x in METRICS},
                    "val_n": m["n"],
                })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out

#expand around finalist checkpoints before full validation sweep
def expand_steps(steps, finalists, radius):
    idx = {s: i for i, s in enumerate(steps)}
    keep = set(finalists)
    for s in finalists:
        i = idx[s]
        keep.update(steps[max(0, i - radius): min(len(steps), i + radius + 1)])
    return sorted(keep)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora-root", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--train-mean", type=float, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k-grid", default="0,1,2,5,8,11")
    ap.add_argument("--temp-grid", default="0")
    ap.add_argument("--selection-metric", default="RMSE", choices=["MAE", "RMSE", "R2"])
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--fp-radius", type=int, default=2)
    ap.add_argument("--fp-nbits", type=int, default=2048)
    ap.add_argument("--context-limit", type=int, default=4096)
    ap.add_argument("--reserved-for-generation", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--triage-radius", type=int, default=2)
    ap.add_argument("--outside-checkpoints", type=int, default=3)
    ap.add_argument("--coarse-k-grid", default="0,5")
    ap.add_argument("--coarse-temp-grid", default="0")
    ap.add_argument("--top-finalists", type=int, default=2)
    ap.add_argument("--expand-radius", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl(args.train_jsonl)
    val_records = load_jsonl(args.val_jsonl)
    test_records = load_jsonl(args.test_jsonl)
    train_fps = build_fp_cache(train_records, radius=args.fp_radius, nbits=args.fp_nbits)

    k_grid = parse_grid(args.k_grid, int)
    temp_grid = parse_grid(args.temp_grid, float)
    coarse_k_grid = parse_grid(args.coarse_k_grid, int)
    coarse_temp_grid = parse_grid(args.coarse_temp_grid, float)

    triage_steps, hist, ckpts = shortlist_steps(args.lora_root, args.triage_radius, args.outside_checkpoints)
    coarse = sweep_steps(args, triage_steps, val_records, train_records, train_fps, coarse_k_grid, coarse_temp_grid)
    ranked = sorted(coarse, key={
        "MAE": lambda r: (r["val_MAE"], r["val_RMSE"], -r["val_R2"], r["step"]),
        "R2": lambda r: (-r["val_R2"], r["val_RMSE"], r["val_MAE"], r["step"]),
        "RMSE": lambda r: (r["val_RMSE"], r["val_MAE"], -r["val_R2"], r["step"]),
    }[args.selection_metric])

    finalists = []
    for r in ranked:
        if r["step"] not in finalists:
            finalists.append(r["step"])
        if len(finalists) >= args.top_finalists:
            break

    all_steps = [int(p.name.split("-")[-1]) for p in ckpts]
    full_steps = expand_steps(all_steps, finalists, args.expand_radius)
    full = sweep_steps(args, full_steps, val_records, train_records, train_fps, k_grid, temp_grid)
    best = choose_best(full, args.selection_metric)

    #evaluate the selected checkpoint and hyperparameters once on the test set
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_model(args.base_model, best["path"], args.load_8bit)
    test_rows = predict_rows(
        model, tokenizer, test_records, train_records, train_fps, best["k"], best["temperature"],
        args.max_new_tokens, args.fp_radius, args.fp_nbits,
        args.context_limit, args.reserved_for_generation,
    )
    test_metrics = compute_metrics(
        [r["output"] for r in test_rows],
        [r["prediction"] for r in test_rows],
        baseline_mean=args.train_mean,
    )

    with open(outdir / "eval_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "eval_loss"])
        writer.writeheader()
        writer.writerows(hist)

    with open(outdir / "coarse_validation_grid.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "step", "k", "temperature", "val_n", "val_MAE", "val_RMSE", "val_R2", "val_R2_test_trainmean", "path"])
        writer.writeheader()
        writer.writerows(coarse)

    with open(outdir / "validation_grid.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "step", "k", "temperature", "val_n", "val_MAE", "val_RMSE", "val_R2", "val_R2_test_trainmean", "path"])
        writer.writeheader()
        writer.writerows(full)

    test_cols = ["id", "smiles", "input", "output", "prediction", "k_requested", "k_used", "temperature"]
    with open(outdir / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=test_cols)
        writer.writeheader()
        writer.writerows([{c: r.get(c) for c in test_cols} for r in test_rows])

    plot_metric(full, "val_RMSE", "RMSE", "Validation RMSE vs k", outdir / "validation_rmse.png")
    plot_metric(full, "val_R2_test_trainmean", r"$\mathbf{R^2_{(test,\ trainmean)}}$", r"Validation $\mathbf{R^2_{(test,\ trainmean)}}$ vs k", outdir / "validation_r2_test_trainmean.png")

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_params": {"checkpoint": best["checkpoint"], "step": best["step"], "k": best["k"], "temperature": best["temperature"]},
            "selection_metric": args.selection_metric,
            "triage_steps": triage_steps,
            "full_steps": full_steps,
            "val_best": {"n": best["val_n"], "MAE": best["val_MAE"], "RMSE": best["val_RMSE"], "R2": best["val_R2"], "R2_test_trainmean": best["val_R2_test_trainmean"]},
            "test": test_metrics,
            "param_grid": {"k_grid": k_grid, "temp_grid": temp_grid, "coarse_k_grid": coarse_k_grid, "coarse_temp_grid": coarse_temp_grid},
            "train_mean_for_R2_test_trainmean": args.train_mean,
            "dataset_sizes": {"train": len(train_records), "val": len(val_records), "test": len(test_records)},
            "inputs_from_launcher": vars(args),
        }, f, indent=2)

    print(json.dumps({"best_checkpoint": best["checkpoint"], "best_k": best["k"], "best_temperature": best["temperature"], "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()