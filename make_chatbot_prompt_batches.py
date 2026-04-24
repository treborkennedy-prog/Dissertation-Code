import os
import pandas as pd
from rdkit import Chem
from rdkit import rdBase

# edit these
path_to_csv = "PATH/TO/input.csv" #columns: Row_ID, canonical_SMILES
out_dir = "PATH/TO/output_prompts"
batch_size = 50
order_seed = 42
base_seed = 12345

PROMPT = """You are a chemistry property predictor.

Task:
#put task instruction here

Input format:
Each molecule is provided as: <id>: <SMILES>

Output format (STRICT):
Return EXACTLY {N} lines.
Each line must be: <id>,<VALUE>
- No header
- No extra text

Molecules:
"""

os.makedirs(out_dir, exist_ok=True)
df = pd.read_csv(path_to_csv)[["Row_ID","canonical_SMILES"]]
df["Row_ID"] = df["Row_ID"].astype(int)
df = df.sample(frac=1, random_state=order_seed).reset_index(drop=True)  # shuffle molecule order

def rand_smiles(smi, rid):
    rdBase.SeedRandomNumberGenerator(base_seed + rid) #fixes randomized SMILES so they are the same every time the script is run
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=False, doRandom=True, isomericSmiles=True)

df["rand_SMILES"] = [rand_smiles(s, rid) for rid, s in zip(df.Row_ID, df.canonical_SMILES)]

def write(run, col):
    d = os.path.join(out_dir, run); os.makedirs(d, exist_ok=True)
    for i in range(0, len(df), batch_size):
        b = df.iloc[i:i+batch_size]
        with open(os.path.join(d, f"{run}_batch_{i//batch_size+1:03d}_n{len(b)}.txt"), "w", encoding="utf-8") as f:
            f.write(PROMPT.format(N=len(b)))
            f.write("\n".join(f"{int(r.Row_ID)}: {getattr(r, col)}" for r in b.itertuples()) + "\n")

write("canonical", "canonical_SMILES")
write("randomized", "rand_SMILES")