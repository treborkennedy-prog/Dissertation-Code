#Adapted from https://github.com/NYUSHCS/MolecularGPT

import re
import json
import csv
import random
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
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

def parse_grid(x, cast):
    xs = [cast(v) for v in x] if isinstance(x, (list, tuple)) else [cast(v.strip()) for v in str(x).split(",") if v.strip()]
    seen, out = set(), []
    for v in xs:
        k = f"{v:.12g}" if isinstance(v, float) else v
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out

#Fix RNGs so CV and generation are reproducible
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#take the final number from the model output
def extract_number(text):
    xs = NUMBER_PATTERN.findall(text)
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

#map each row-ID to its split group for structure-aware splitting
def load_group_ids(path):
    out = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[str(row["Row-ID"])] = str(row["benchmark_group_id"])
    return out

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
            return prompt
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
        prompt = fit_prompt(tokenizer, row["instruction"], examples, row["input"], context_limit, reserved_for_generation)
        enc = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            out = model.generate(**enc, generation_config=gen_cfg)
        pred = extract_number(tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True))
        rows.append({"id": row["id"], "input": row["input"], "output": float(row["output"]), "prediction": pred})
    return rows

#select best hyperparameter config
def choose_best(rows, metric):
    metric = metric.upper()

    def key(r):
        if metric == "MAE":
            return (r["mean_MAE"], r["mean_RMSE"], -r["mean_R2"], r["k"], r["temperature"])
        if metric == "R2":
            return (-r["mean_R2"], r["mean_RMSE"], r["mean_MAE"], r["k"], r["temperature"])
        return (r["mean_RMSE"], r["mean_MAE"], -r["mean_R2"], r["k"], r["temperature"])

    return min(rows, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora-weights", default=None)
    ap.add_argument("--dev-jsonl", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--split-labels", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--k-grid", default="0,1,2,5,8,11")
    ap.add_argument("--temp-grid", default="0")
    ap.add_argument("--selection-metric", default="RMSE", choices=["MAE", "RMSE", "R2"])
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--fp-radius", type=int, default=2)
    ap.add_argument("--fp-nbits", type=int, default=2048)
    ap.add_argument("--context-limit", type=int, default=4096)
    ap.add_argument("--reserved-for-generation", type=int, default=64)
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    lora_weights = None if str(args.lora_weights).strip().lower() in {"", "none", "null"} else args.lora_weights
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_csv.with_name(out_csv.stem + "__summary.json")

    dev_records = load_jsonl(args.dev_jsonl)
    test_records = load_jsonl(args.test_jsonl)
    group_map = load_group_ids(args.split_labels)
    dev_groups = np.asarray([group_map[r["id"]] for r in dev_records], dtype=object)
    dev_fps = build_fp_cache(dev_records, radius=args.fp_radius, nbits=args.fp_nbits)

    #load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    #Load the base model, optionally with LoRA weights
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        load_in_8bit=args.load_8bit,
        device_map={"": 0},
        attn_implementation="eager",
    )
    if lora_weights is not None:
        model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16, device_map={"": 0})
    model.eval()

    #set up the k/temperature sweep and grouped CV
    k_grid = parse_grid(args.k_grid, int)
    temp_grid = parse_grid(args.temp_grid, float)
    splitter = GroupKFold(n_splits=args.cv_folds)
    x_dummy = np.arange(len(dev_records))
    dev_cv = []

    for temperature in temp_grid:
        for k in k_grid:
            fold_metrics = []
            for fold, (tr_idx, va_idx) in enumerate(splitter.split(x_dummy, groups=dev_groups), start=1):
                set_seed(args.seed + int(k) * 1000 + fold)
                train_records = [dev_records[i] for i in tr_idx]
                val_records = [dev_records[i] for i in va_idx]
                train_fps = [dev_fps[i] for i in tr_idx]
                pred_rows = predict_rows(
                    model, tokenizer, val_records, train_records, train_fps, k, temperature,
                    args.max_new_tokens, args.fp_radius, args.fp_nbits,
                    args.context_limit, args.reserved_for_generation,
                )
                fold_metrics.append({
                    "fold": fold,
                    **compute_metrics(
                        [r["output"] for r in pred_rows],
                        [r["prediction"] for r in pred_rows],
                        baseline_mean=float(np.mean([r["output"] for r in train_records])),
                    ),
                })

            #store metrics for hyperparameter config
            dev_cv.append({
                "k": int(k),
                "temperature": float(temperature),
                "fold_metrics": fold_metrics,
                "mean_MAE": float(np.mean([m["MAE"] for m in fold_metrics])),
                "std_MAE": float(np.std([m["MAE"] for m in fold_metrics])),
                "mean_RMSE": float(np.mean([m["RMSE"] for m in fold_metrics])),
                "std_RMSE": float(np.std([m["RMSE"] for m in fold_metrics])),
                "mean_R2": float(np.mean([m["R2"] for m in fold_metrics])),
                "std_R2": float(np.std([m["R2"] for m in fold_metrics])),
                "mean_R2_test_trainmean": float(np.mean([m["R2_test_trainmean"] for m in fold_metrics])),
                "std_R2_test_trainmean": float(np.std([m["R2_test_trainmean"] for m in fold_metrics])),
            })

    #select best CV config, then evaluate on heldout test set 
    best = choose_best(dev_cv, args.selection_metric)
    best_k, best_temp = int(best["k"]), float(best["temperature"])
    set_seed(args.seed + 999999)

    test_rows = predict_rows(
        model, tokenizer, test_records, dev_records, dev_fps, best_k, best_temp,
        args.max_new_tokens, args.fp_radius, args.fp_nbits,
        args.context_limit, args.reserved_for_generation,
    )
    test_metrics = compute_metrics(
        [r["output"] for r in test_rows],
        [r["prediction"] for r in test_rows],
        baseline_mean=float(np.mean([r["output"] for r in dev_records])),
    )

    #save test set predictions
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "input", "output", "prediction"])
        writer.writeheader()
        writer.writerows(test_rows)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "best_params": {"k": best_k, "temperature": best_temp},
            "selection_metric": args.selection_metric,
            "param_grid": {"k_grid": k_grid, "temp_grid": temp_grid},
            "dev_cv": dev_cv,
            "test": test_metrics,
            "inputs_from_launcher": vars(args),
        }, f, indent=2)


if __name__ == "__main__":
    main()