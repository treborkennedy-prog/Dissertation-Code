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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold


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
    return desc, fp, combined

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
    desc, fp, combined = build_feature_sets(df)
    keep = np.isfinite(desc.to_numpy()).all(axis=1) & np.isfinite(fp.to_numpy()).all(axis=1) & np.isfinite(combined.to_numpy()).all(axis=1)
    df = df.loc[keep].reset_index(drop=True)
    desc = desc.loc[keep].reset_index(drop=True)
    fp = fp.loc[keep].reset_index(drop=True)
    combined = combined.loc[keep].reset_index(drop=True)
    y = y[keep]
    return df, desc, fp, combined, y

#hyperparameter grids
def grid_from_name(name):
    if name == "stage1":
        return {
            "feature_set": ["desc", "fp", "combined"],
            "et__n_estimators": [100, 1000],
            "et__max_depth": [None, 30],
            "et__min_samples_split": [2, 4],
            "et__min_samples_leaf": [1, 2],
            "et__max_features": [1.0, 0.5, "sqrt"],
        }
    if name == "pKa_stage2":
        return {
            "feature_set": ["combined"],
            "et__n_estimators": [100, 200, 300],
            "et__max_depth": [None, 40],
            "et__min_samples_split": [2, 3],
            "et__min_samples_leaf": [2, 3, 4],
            "et__max_features": [0.3, 0.4, 0.5, 0.6, 0.7],
        }
    if name == "logS_stage2":
        return {
            "feature_set": ["combined"],
            "et__n_estimators": [1000, 1500],
            "et__max_depth": [20, 30, 40],
            "et__min_samples_split": [2, 3],
            "et__min_samples_leaf": [2, 3],
            "et__max_features": [1.0, 0.8, 0.6],
        }

    raise ValueError(name)


def drop_zero_var(X_train, X_other):
    keep = X_train.std(axis=0) > 0
    return X_train[:, keep], X_other[:, keep], keep

#evaluate hyperparameter configuration
def evaluate_config(X_dev, y_dev, groups, params, n_splits, cv_mode, n_jobs, random_state):
    maes = []
    rmses = []
    r2s = []
    if cv_mode == "group":
        splits = GroupKFold(n_splits=n_splits).split(X_dev, y_dev, groups)
    else:
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X_dev, y_dev)
    for tr, va in splits:
        X_tr, X_va, _ = drop_zero_var(X_dev[tr], X_dev[va])
        model = ExtraTreesRegressor(
            n_estimators=int(params["et__n_estimators"]),
            max_depth=None if params["et__max_depth"] is None else int(params["et__max_depth"]),
            min_samples_split=int(params["et__min_samples_split"]),
            min_samples_leaf=int(params["et__min_samples_leaf"]),
            max_features=params["et__max_features"],
            random_state=random_state,
            n_jobs=n_jobs,
        )
        model.fit(X_tr, y_dev[tr])
        pred = model.predict(X_va)
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
    ap.add_argument("--cv-mode", default="group", choices=["group", "random"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--top-n-features", type=int, default=200)
    ap.add_argument("--launcher", default="")
    ap.add_argument("--script-dir", default="")
    ap.add_argument("--script-path", default="")
    ap.add_argument("--conda-env", default="")
    ap.add_argument("--module-load", default="")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    df, desc_df, fp_df, combined_df, y = load_data(args.dataset, args.csv, args.split_labels)
    feature_frames = {"desc": desc_df, "fp": fp_df, "combined": combined_df}
    feature_names_map = {"desc": desc_df.columns.to_numpy(), "fp": fp_df.columns.to_numpy(), "combined": combined_df.columns.to_numpy()}
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
        rows.append(evaluate_config(X_dev, y_dev, g_dev, params, args.n_splits, args.cv_mode, args.n_jobs, args.random_state))

    #rank configurations by RMSE, then MAE, then R2
    cv_results = pd.DataFrame(rows)
    cv_results = cv_results.sort_values(["cv_RMSE_mean", "cv_MAE_mean", "cv_R2_mean"], ascending=[True, True, False]).reset_index(drop=True)
    cv_results.to_csv(os.path.join(args.outdir, "cv_results.csv"), index=False)

    #select best configuration
    best = cv_results.iloc[0].to_dict()
    best_params = {
        "feature_set": best["feature_set"],
        "et__n_estimators": int(best["et__n_estimators"]),
        "et__max_depth": None if pd.isna(best["et__max_depth"]) else int(best["et__max_depth"]),
        "et__min_samples_split": int(best["et__min_samples_split"]),
        "et__min_samples_leaf": int(best["et__min_samples_leaf"]),
        "et__max_features": best["et__max_features"],
    }

    #Retrain model on entire development set, then evaluate on test set
    X_dev = feature_frames[best_params["feature_set"]].loc[dev_mask].to_numpy()
    X_test = feature_frames[best_params["feature_set"]].loc[test_mask].to_numpy()
    feature_names = feature_names_map[best_params["feature_set"]]
    X_dev, X_test, keep = drop_zero_var(X_dev, X_test)
    feature_names = feature_names[keep]
    model = ExtraTreesRegressor(
        n_estimators=int(best_params["et__n_estimators"]),
        max_depth=best_params["et__max_depth"],
        min_samples_split=int(best_params["et__min_samples_split"]),
        min_samples_leaf=int(best_params["et__min_samples_leaf"]),
        max_features=best_params["et__max_features"],
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    model.fit(X_dev, y_dev)
    y_pred = model.predict(X_test)
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

    fi = pd.DataFrame({"feature_name": feature_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False).reset_index(drop=True)
    fi.head(int(args.top_n_features)).to_csv(os.path.join(args.outdir, "feature_importances.csv"), index=False)
    top_feature_names = fi.head(int(args.top_n_features))["feature_name"].astype(str).tolist()

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
            "n_jobs": int(args.n_jobs),
            "random_state": int(args.random_state),
            "top_n_features": int(args.top_n_features),
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
        "top_feature_names": top_feature_names,
        "extra_outputs": {
            "cv_results_csv": os.path.join(args.outdir, "cv_results.csv"),
            "test_predictions_csv": os.path.join(args.outdir, "test_predictions.csv"),
            "feature_importances_csv": os.path.join(args.outdir, "feature_importances.csv"),
        },
        "runtime_minutes": float((time.time() - t0) / 60.0),
    }

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()