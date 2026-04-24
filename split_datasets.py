import json
import random
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Cluster import Butina

fpgen = GetMorganGenerator(radius=2, fpSize=2048)

#fingerprint creation
def fp(s):
    return fpgen.GetFingerprint(Chem.MolFromSmiles(s))

#scaffold assignment
def murcko(s):
    x = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s), includeChirality=False)
    return x if x else "__NO_SCAFFOLD__"

#butina cluster assignment
def butina_from_fps(fps, cutoff=0.60):
    n = len(fps)
    if n <= 1:
        return [list(range(n))]
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - x for x in sims])
    return [list(c) for c in Butina.ClusterData(np.array(dists, dtype=np.float32), n, cutoff, isDistData=True)]

#hybrid method for big scaffolds
def subclusters(fps, idxs, cutoff=0.60):
    local = butina_from_fps([fps[i] for i in idxs], cutoff)
    return [[idxs[j] for j in g] for g in local]

#assign groups to the test set
def choose_test(groups, y, target_frac=0.2, n_seed_trials=1000):
    y = np.asarray(y, dtype=float)
    n = len(y)
    target_n = target_frac * n
    mu = float(np.mean(y))
    sd = float(np.std(y, ddof=1))
    qs = [float(np.quantile(y, q)) for q in (0.1, 0.5, 0.9)]
    best = None

    for seed in range(n_seed_trials):
        order = list(range(len(groups)))
        random.Random(seed).shuffle(order)

        chosen = []
        size = 0

        for gi in order:
            if abs((size + len(groups[gi])) - target_n) < abs(size - target_n):
                chosen.append(gi)
                size += len(groups[gi])

        #make sure test set is at least 19% of dataset
        if size < 0.19 * n:
            rest = [gi for gi in order if gi not in chosen]
            rest.sort(key=lambda gi: abs((size + len(groups[gi])) - target_n))
            for gi in rest:
                chosen.append(gi)
                size += len(groups[gi])
                if size >= 0.19 * n:
                    break

        test_idx = np.array(sorted([i for gi in chosen for i in groups[gi]]), dtype=int)
        test_y = y[test_idx]
        frac = len(test_idx) / n

        #score the split, test set should try to match full dataset in mean, stdev, quantiles
        score = abs(frac - target_frac) / 0.02
        score += abs(float(np.mean(test_y)) - mu) / sd
        score += abs(float(np.std(test_y, ddof=1)) - sd) / sd
        for q, ref in zip((0.1, 0.5, 0.9), qs):
            score += abs(float(np.quantile(test_y, q)) - ref) / sd

        if best is None or score < best[0]:
            best = (score, seed, test_idx)

    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pka", "logs"], required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if args.dataset == "pka":
        smiles_col = "Smiles_canonical"
        y_col = "pKa_num"
        smiles = df[smiles_col].astype(str).tolist()
        fps = [fp(s) for s in smiles]

        scaffold_map = defaultdict(list)
        for i, s in enumerate(smiles):
            scaffold_map[murcko(s)].append(i)

        groups = []
        for _, idxs in scaffold_map.items():
            if len(idxs) > 300:
                groups.extend(subclusters(fps, idxs, 0.60))
            else:
                groups.append(idxs)

    else:
        smiles_col = "Canonical_SMILES"
        y_col = "LogS exp (mol/L)"
        smiles = df[smiles_col].astype(str).tolist()
        groups = butina_from_fps([fp(s) for s in smiles], 0.55)

    score, seed, test_idx = choose_test(groups, df[y_col].to_numpy())

    split = np.array(["dev"] * len(df), dtype=object)
    split[test_idx] = "test"

    #assign a group ID to every molecule
    group_id = np.empty(len(df), dtype=object)
    for gi, g in enumerate(groups):
        gid = f"{args.dataset}_group_{gi:04d}"
        for i in g:
            group_id[i] = gid

    out = pd.DataFrame({
        "Row-ID": df["Row-ID"],
        "benchmark_split": split,
        "benchmark_group_id": group_id,
    })    

    out.to_csv(args.out_csv, index=False)

    print(json.dumps({
        "seed_for_partition_search": seed,
        "n_test": int((split == "test").sum())
    }, indent=2))

if __name__ == "__main__":
    main()