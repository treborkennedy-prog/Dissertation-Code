import os
import json
import math
import time
import argparse
import itertools
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def parse_mol(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return mol

def calc_desc(mol):
    return Descriptors.CalcMolDescriptors(mol, missingVal=np.nan)

def build_feature_sets(df):
    desc = pd.DataFrame([calc_desc(m) for m in df["mol"]], index=df.index).replace([np.inf, -np.inf], np.nan)
    desc = desc.drop(columns=desc.columns[desc.isna().any()].tolist()).astype(float)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096, includeChirality=False)
    fps = []
    for mol in df["mol"]:
        arr = np.zeros(4096, dtype=np.uint8)
        ConvertToNumpyArray(fpgen.GetFingerprint(mol), arr)
        fps.append(arr)
    fp = pd.DataFrame(np.vstack(fps), columns=[f"fp_{i}" for i in range(4096)], index=df.index).astype(float)
    combined = pd.concat([desc, fp], axis=1)
    return desc, combined

def load_data(dataset, csv_path, split_path):
    df = pd.read_csv(csv_path)
    splits = pd.read_csv(split_path)
    df = df.merge(splits[["Row-ID", "benchmark_split", "benchmark_group_id"]], on="Row-ID", how="inner")
    if dataset == "pka":
        df = df.rename(columns={"Smiles_canonical": "smiles", "pKa_num": "y"})
    else:
        df = df.rename(columns={"Canonical_SMILES": "smiles", "LogS exp (mol/L)": "y"})
    df = df.dropna(subset=["smiles", "y", "benchmark_split", "benchmark_group_id"]).reset_index(drop=True)
    df["benchmark_split"] = df["benchmark_split"].astype(str).str.lower().str.strip()
    df["mol"] = [parse_mol(s) for s in df["smiles"]]
    df = df.dropna(subset=["mol"]).reset_index(drop=True)
    y = pd.to_numeric(df["y"], errors="coerce").astype(float).values
    keep = np.isfinite(y)
    df = df.loc[keep].reset_index(drop=True)
    y = y[keep]
    desc, combined = build_feature_sets(df)
    keep = np.isfinite(desc.to_numpy()).all(axis=1) & np.isfinite(combined.to_numpy()).all(axis=1)
    df = df.loc[keep].reset_index(drop=True)
    desc = desc.loc[keep].reset_index(drop=True)
    combined = combined.loc[keep].reset_index(drop=True)
    y = y[keep]
    return df, desc, combined, y

#hyperparameter grids
def grid_from_name(name):
    if name == "stage1_coarse":
        return {
            "feature_set": ["desc", "combined"],
            "mlp__hidden_layer_sizes": [(100,), (200,), (200, 200)],
            "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "mlp__learning_rate_init": [1e-3, 3e-4],
            "mlp__early_stopping": [True, False],
        }
    if name == "pka_stage2":
        return {
            "feature_set": ["desc"],
            "mlp__hidden_layer_sizes": [(150,), (200,), (250,), (300,)],
            "mlp__alpha": [0.003, 0.01, 0.03, 0.1],
            "mlp__learning_rate_init": [0.0001, 0.0003, 0.0005],
            "mlp__early_stopping": [False, True],
        }
    if name == "pka_stage3":
        return {
            "feature_set": ["desc"],
            "mlp__hidden_layer_sizes": [(200,), (250,), (300,)],
            "mlp__alpha": [0.03, 0.1, 0.3],
            "mlp__learning_rate_init": [0.0003, 0.0005, 0.0007, 0.001],
            "mlp__early_stopping": [True],
        }
    if name == "logS_stage2":
        return {
            "feature_set": ["desc"],
            "mlp__hidden_layer_sizes": [(50,), (100,), (150,), (200,)],
            "mlp__alpha": [1e-6, 3e-6, 1e-5, 3e-5],
            "mlp__learning_rate_init": [1e-4, 2e-4, 3e-4, 5e-4],
            "mlp__early_stopping": [True],
        }
    if name == "logS_stage3":
        return {
            "feature_set": ["desc"],
            "mlp__hidden_layer_sizes": [(75,), (100,), (125,), (150,), (100,100)],
            "mlp__alpha": [6e-6, 1e-5, 2e-5],
            "mlp__learning_rate_init": [1.5e-4, 2e-4, 2.5e-4],
            "mlp__early_stopping": [True],
        }
       
       
    raise ValueError(name)

#drop zero variance features and feature scaling
def transform_block(X_train, X_other):
    keep = X_train.std(axis=0) > 0
    X_train = X_train[:, keep]
    X_other = X_other[:, keep]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_other = scaler.transform(X_other)
    return X_train, X_other, keep

def make_model(params, random_state):
    return MLPRegressor(
        hidden_layer_sizes=params["mlp__hidden_layer_sizes"],
        alpha=float(params["mlp__alpha"]),
        learning_rate_init=float(params["mlp__learning_rate_init"]),
        early_stopping=bool(params["mlp__early_stopping"]),
        max_iter=1000,
        random_state=random_state
    )

#evaluate hyperparameter configuration
def evaluate_config(X_dev, y_dev, groups, params, n_splits, cv_mode, random_state):
    maes = []
    rmses = []
    r2s = []
    if cv_mode == "group":
        splits = GroupKFold(n_splits=n_splits).split(X_dev, y_dev, groups)
    else:
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X_dev, y_dev)
    for tr, va in splits:
        X_tr2, X_va2, _ = transform_block(X_dev[tr], X_dev[va])
        model = make_model(params, random_state)
        model.fit(X_tr2, y_dev[tr])
        pred = model.predict(X_va2)
        maes.append(float(mean_absolute_error(y_dev[va], pred)))
        rmses.append(float(math.sqrt(mean_squared_error(y_dev[va], pred))))
        r2s.append(float(r2_score(y_dev[va], pred)))
    return {
        **params,
        "cv_MAE_mean": float(np.mean(maes)),
        "cv_MAE_sd": float(np.std(maes, ddof=1)),
        "cv_RMSE_mean": float(np.mean(rmses)),
        "cv_RMSE_sd": float(np.std(rmses, ddof=1)),
        "cv_R2_mean": float(np.mean(r2s)),
        "cv_R2_sd": float(np.std(r2s, ddof=1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pka", "logs"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split-labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--cv-mode", default="group", choices=["group", "random"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--launcher", default="")
    ap.add_argument("--script-dir", default="")
    ap.add_argument("--script-path", default="")
    ap.add_argument("--conda-env", default="")
    ap.add_argument("--module-load", default="")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    df, desc_df, combined_df, y = load_data(args.dataset, args.csv, args.split_labels)
    feature_frames = {"desc": desc_df, "combined": combined_df}
    dev_mask = df["benchmark_split"].eq("dev").values
    test_mask = df["benchmark_split"].eq("test").values
    y_dev = y[dev_mask]
    y_test = y[test_mask]
    g_dev = df.loc[dev_mask, "benchmark_group_id"].astype(str).values
    df_test = df.loc[test_mask].reset_index(drop=True).copy()

    #sweep hyperparameter grid
    grid = grid_from_name(args.grid)
    rows = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        X_dev = feature_frames[params["feature_set"]].loc[dev_mask].to_numpy()
        rows.append(evaluate_config(X_dev, y_dev, g_dev, params, args.n_splits, args.cv_mode, args.random_state))

    #rank configurations by RMSE, then MAE, then R2
    cv_results = pd.DataFrame(rows)
    cv_results = cv_results.sort_values(["cv_RMSE_mean", "cv_MAE_mean", "cv_R2_mean"], ascending=[True, True, False]).reset_index(drop=True)
    cv_results.to_csv(os.path.join(args.outdir, "cv_results.csv"), index=False)

    #select best configuration
    best = cv_results.iloc[0].to_dict()
    best_params = {
        "feature_set": best["feature_set"],
        "mlp__hidden_layer_sizes": best["mlp__hidden_layer_sizes"],
        "mlp__alpha": float(best["mlp__alpha"]),
        "mlp__learning_rate_init": float(best["mlp__learning_rate_init"]),
        "mlp__early_stopping": bool(best["mlp__early_stopping"]),
    }

    #Retrain model on entire development set, then evaluate on test set
    X_dev = feature_frames[best_params["feature_set"]].loc[dev_mask].to_numpy()
    X_test = feature_frames[best_params["feature_set"]].loc[test_mask].to_numpy()
    X_dev2, X_test2, keep = transform_block(X_dev, X_test)
    model = make_model(best_params, args.random_state)
    model.fit(X_dev2, y_dev)
    y_pred = model.predict(X_test2)

    test_metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "R2_test_trainmean": float(1.0 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_dev)) ** 2)),
    }

    #save test set predictions
    pred_df = df_test[["Row-ID", "smiles", "benchmark_group_id"]].copy()
    pred_df["true_value"] = y_test
    pred_df["pred_value"] = y_pred
    pred_df["residual"] = pred_df["true_value"] - pred_df["pred_value"]
    pred_df["abs_error"] = np.abs(pred_df["residual"])
    pred_df.to_csv(os.path.join(args.outdir, "test_predictions.csv"), index=False)

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
            "conda_env": args.conda_env,
            "module_load": args.module_load,
        },
        "n_total": int(len(df)),
        "n_dev": int(dev_mask.sum()),
        "n_test": int(test_mask.sum()),
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
        "final_feature_count_before_zero_variance": int(X_dev.shape[1]),
        "final_feature_count_after_zero_variance": int(keep.sum()),
        "extra_outputs": {
            "cv_results_csv": os.path.join(args.outdir, "cv_results.csv"),
            "test_predictions_csv": os.path.join(args.outdir, "test_predictions.csv"),
        },
        "runtime_minutes": float((time.time() - t0) / 60.0),
    }

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()