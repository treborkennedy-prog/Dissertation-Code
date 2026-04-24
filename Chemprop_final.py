import os
import json
import math
import time
import argparse
import itertools
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

#standardise column names for both datasets
def load_data(dataset, csv_path, split_path):
    df = pd.read_csv(csv_path)
    splits = pd.read_csv(split_path)
    df = df.merge(splits[["Row-ID", "benchmark_split", "benchmark_group_id"]], on="Row-ID", how="inner")
    if dataset == "pka":
        df = df.rename(columns={"Smiles_canonical": "smiles", "pKa_num": "y"})
    else:
        df = df.rename(columns={"Canonical_SMILES": "smiles", "LogS exp (mol/L)": "y"})
    df["benchmark_split"] = df["benchmark_split"].astype(str).str.lower().str.strip()
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["Row-ID", "smiles", "y", "benchmark_split", "benchmark_group_id"]).reset_index(drop=True)
    return df

#hyperparameter grids
def grid_from_name(name):
    if name == "stage1":
        return {
            "depth": [3, 4, 5],
            "message_hidden_dim": [300, 600],
            "ffn_hidden_dim": [300, 600],
            "ffn_num_layers": [1, 2],
            "dropout": [0.0, 0.1],
        }
    if name == "pKa_stage2":
        return {
            "depth": [2, 3, 4],
            "message_hidden_dim": [600, 800],
            "ffn_hidden_dim": [600, 800],
            "ffn_num_layers": [2, 3],
            "dropout": [0.05, 0.10, 0.15],
        }
    if name == "logS_stage2":
        return {
            "depth": [2, 3, 4],
            "message_hidden_dim": [200, 250, 300, 350, 400],
            "ffn_hidden_dim": [300, 600, 800],
            "ffn_num_layers": [2, 3],
            "dropout": [0.0, 0.05, 0.1],
        }
    raise ValueError(name)

def make_splits(n, groups, n_splits, cv_mode, random_state):
    x = np.zeros((n, 1))
    y = np.zeros(n)
    if cv_mode == "group":
        splitter = GroupKFold(n_splits=n_splits).split(x, y, groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(x, y)
    return [{"train": tr.tolist(), "val": va.tolist()} for tr, va in splitter]

#run a  Chemprop command and stop the script if it fails
def run_cmd(cmd):
    subprocess.run(cmd, check=True)

#find the Chemprop prediction column
def infer_pred_column(input_df, pred_df):
    extra = [c for c in pred_df.columns if c not in input_df.columns]
    return extra[0]

#train Chemprop for one CV fold
def eval_one_fold(dev_df, fold_split, params, fold_dir, args):
    os.makedirs(fold_dir, exist_ok=True)
    fold_df = dev_df.copy()
    fold_df["split"] = ""
    fold_df.loc[fold_split["train"], "split"] = "train"
    fold_df.loc[fold_split["val"], "split"] = "val"
    fold_csv = os.path.join(fold_dir, "fold.csv")
    fold_df.to_csv(fold_csv, index=False)
    val_df = dev_df.iloc[fold_split["val"]].reset_index(drop=True).copy()
    val_infer = val_df.drop(columns=["y"])
    val_infer_csv = os.path.join(fold_dir, "val_infer.csv")
    val_infer.to_csv(val_infer_csv, index=False)
    raw_pred_csv = os.path.join(fold_dir, "val_predictions_raw.csv")
    train_cmd = [
        "chemprop", "train",
        "--data-path", fold_csv,
        "--task-type", "regression",
        "--output-dir", fold_dir,
        "--logfile", os.path.join(fold_dir, "chemprop_train.log"),
        "--smiles-columns", "smiles",
        "--target-columns", "y",
        "--metrics", "mae", "rmse", "r2",
        "--tracking-metric", "rmse",
        "--splits-column", "split",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--accelerator", args.accelerator,
        "--devices", str(args.devices),
        "--depth", str(params["depth"]),
        "--message-hidden-dim", str(params["message_hidden_dim"]),
        "--ffn-hidden-dim", str(params["ffn_hidden_dim"]),
        "--ffn-num-layers", str(params["ffn_num_layers"]),
        "--dropout", str(params["dropout"]),
    ]
    if args.num_workers is not None:
        train_cmd += ["--num-workers", str(args.num_workers)]
    if args.patience is not None:
        train_cmd += ["--patience", str(args.patience)]
    run_cmd(train_cmd)
    pred_cmd = [
        "chemprop", "predict",
        "--test-path", val_infer_csv,
        "--model-path", fold_dir,
        "--preds-path", raw_pred_csv,
        "--smiles-columns", "smiles",
        "--accelerator", args.accelerator,
        "--devices", str(args.devices),
    ]
    if args.num_workers is not None:
        pred_cmd += ["--num-workers", str(args.num_workers)]
    run_cmd(pred_cmd)
 
    #calculate metrics for validation set
    pred_df = pd.read_csv(raw_pred_csv)
    pred_col = infer_pred_column(val_infer, pred_df)
    pred_sub = pred_df[["Row-ID", pred_col]].rename(columns={pred_col: "pred_value"})
    merged = val_df.merge(pred_sub, on="Row-ID", how="inner")
    y_true = merged["y"].astype(float).to_numpy()
    y_pred = pd.to_numeric(merged["pred_value"], errors="coerce").astype(float).to_numpy()
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

#evaluate a hyperparameter configuration across all CV folds
def evaluate_config(dev_df, splits, params, cfg_idx, workdir, args):
    maes = []
    rmses = []
    r2s = []
    for fold_idx, fold_split in enumerate(splits):
        fold_dir = os.path.join(workdir, f"cfg_{cfg_idx:03d}", f"fold_{fold_idx}")
        m = eval_one_fold(dev_df, fold_split, params, fold_dir, args)
        maes.append(m["MAE"])
        rmses.append(m["RMSE"])
        r2s.append(m["R2"])
    return {
        **params,
        "cv_MAE_mean": float(np.mean(maes)),
        "cv_MAE_sd": float(np.std(maes, ddof=1)),
        "cv_RMSE_mean": float(np.mean(rmses)),
        "cv_RMSE_sd": float(np.std(rmses, ddof=1)),
        "cv_R2_mean": float(np.mean(r2s)),
        "cv_R2_sd": float(np.std(r2s, ddof=1)),
    }

#retrain Chemprop on full development set, then evaluate on heldout test set
def final_fit_and_test(dev_df, test_df, params, outdir, args):
    final_dir = os.path.join(outdir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    dev_csv = os.path.join(final_dir, "dev.csv")
    test_infer_csv = os.path.join(final_dir, "test_infer.csv")
    final_splits_json = os.path.join(final_dir, "final_splits.json")
    raw_pred_csv = os.path.join(final_dir, "test_predictions_raw.csv")
    dev_df.to_csv(dev_csv, index=False)
    test_df.drop(columns=["y"]).to_csv(test_infer_csv, index=False)
    with open(final_splits_json, "w") as f:
        json.dump([{"train": list(range(len(dev_df)))}], f)
    train_cmd = [
        "chemprop", "train",
        "--data-path", dev_csv,
        "--task-type", "regression",
        "--output-dir", final_dir,
        "--logfile", os.path.join(final_dir, "chemprop_train.log"),
        "--smiles-columns", "smiles",
        "--target-columns", "y",
        "--metrics", "mae", "rmse", "r2",
        "--splits-file", final_splits_json,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--accelerator", args.accelerator,
        "--devices", str(args.devices),
        "--depth", str(params["depth"]),
        "--message-hidden-dim", str(params["message_hidden_dim"]),
        "--ffn-hidden-dim", str(params["ffn_hidden_dim"]),
        "--ffn-num-layers", str(params["ffn_num_layers"]),
        "--dropout", str(params["dropout"]),
    ]
    if args.num_workers is not None:
        train_cmd += ["--num-workers", str(args.num_workers)]
    run_cmd(train_cmd)
    pred_cmd = [
        "chemprop", "predict",
        "--test-path", test_infer_csv,
        "--model-path", final_dir,
        "--preds-path", raw_pred_csv,
        "--smiles-columns", "smiles",
        "--accelerator", args.accelerator,
        "--devices", str(args.devices),
    ]
    if args.num_workers is not None:
        pred_cmd += ["--num-workers", str(args.num_workers)]
    run_cmd(pred_cmd)
    
    #calculate metrics on test set 
    pred_df = pd.read_csv(raw_pred_csv)
    pred_col = infer_pred_column(test_df.drop(columns=["y"]), pred_df)
    pred_sub = pred_df[["Row-ID", pred_col]].rename(columns={pred_col: "pred_value"})
    merged = test_df.merge(pred_sub, on="Row-ID", how="inner")
    y_true = merged["y"].astype(float).to_numpy()
    y_pred = pd.to_numeric(merged["pred_value"], errors="coerce").astype(float).to_numpy()
    metrics = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "R2_test_trainmean": float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(dev_df["y"].astype(float).to_numpy())) ** 2)),
    }
    pred_out = merged[["Row-ID", "smiles", "benchmark_group_id"]].copy()
    pred_out["true_value"] = y_true
    pred_out["pred_value"] = y_pred
    pred_out["residual"] = pred_out["true_value"] - pred_out["pred_value"]
    pred_out["abs_error"] = np.abs(pred_out["residual"])
    pred_out.to_csv(os.path.join(outdir, "test_predictions.csv"), index=False)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pka", "logs"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split-labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--cv-mode", default="group", choices=["group", "random"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--devices", default="1")
    ap.add_argument("--launcher", default="")
    ap.add_argument("--script-dir", default="")
    ap.add_argument("--script-path", default="")
    ap.add_argument("--conda-env", default="")
    ap.add_argument("--module-load", default="")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    df = load_data(args.dataset, args.csv, args.split_labels)
    dev_df = df.loc[df["benchmark_split"].eq("dev")].reset_index(drop=True).copy()
    test_df = df.loc[df["benchmark_split"].eq("test")].reset_index(drop=True).copy()

    #build CV folds from development set
    groups = dev_df["benchmark_group_id"].astype(str).to_numpy()
    splits = make_splits(len(dev_df), groups, args.n_splits, args.cv_mode, args.random_state)
    grid = grid_from_name(args.grid)
    workdir = os.path.join(args.outdir, "chemprop_runs")
    rows = []
    for cfg_idx, combo in enumerate(itertools.product(*grid.values())):
        params = dict(zip(grid.keys(), combo))
        rows.append(evaluate_config(dev_df, splits, params, cfg_idx, workdir, args))

    #rank configs by RMSE, then MAE, then R2
    cv_results = pd.DataFrame(rows).sort_values(["cv_RMSE_mean", "cv_MAE_mean", "cv_R2_mean"], ascending=[True, True, False]).reset_index(drop=True)
    best = cv_results.iloc[0].to_dict()
    best_params = {
        "depth": int(best["depth"]),
        "message_hidden_dim": int(best["message_hidden_dim"]),
        "ffn_hidden_dim": int(best["ffn_hidden_dim"]),
        "ffn_num_layers": int(best["ffn_num_layers"]),
        "dropout": float(best["dropout"]),
    }
    test_metrics = final_fit_and_test(dev_df, test_df, best_params, args.outdir, args)
    summary = {
        "selection_metric": "RMSE",
        "dataset": args.dataset,
        "inputs_from_launcher": {
            "launcher": args.launcher,
            "script_dir": args.script_dir,
            "script_path": args.script_path,
            "csv": args.csv,
            "split_labels": args.split_labels,
            "outdir": args.outdir,
            "grid": args.grid,
            "cv_mode": args.cv_mode,
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "num_workers": None if args.num_workers is None else int(args.num_workers),
            "patience": None if args.patience is None else int(args.patience),
            "accelerator": args.accelerator,
            "devices": str(args.devices),
            "conda_env": args.conda_env,
            "module_load": args.module_load,
        },
        "n_total": int(len(df)),
        "n_dev": int(len(dev_df)),
        "n_test": int(len(test_df)),
        "best_params": best_params,
        "param_grid_swept": grid,
        "best_dev_cv_metrics": {
            "MAE_mean": float(best["cv_MAE_mean"]),
            "MAE_sd": float(best["cv_MAE_sd"]),
            "RMSE_mean": float(best["cv_RMSE_mean"]),
            "RMSE_sd": float(best["cv_RMSE_sd"]),
            "R2_mean": float(best["cv_R2_mean"]),
            "R2_sd": float(best["cv_R2_sd"]),
        },
        "test_metrics": test_metrics,
        "all_dev_cv_results": cv_results.to_dict(orient="records"),
        "extra_outputs": {
            "test_predictions_csv": os.path.join(args.outdir, "test_predictions.csv"),
            "chemprop_runs_dir": workdir,
            "final_model_dir": os.path.join(args.outdir, "final_model"),
        },
        "runtime_minutes": float((time.time() - t0) / 60.0),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()