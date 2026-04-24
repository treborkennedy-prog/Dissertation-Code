import csv, time
from statistics import mean
from chembl_webresource_client.new_client import new_client
from requests.exceptions import RequestException
from rdkit import Chem

OUT = "chembl_carboxylic_acids_pka.csv"

mols = new_client.molecule
acts = new_client.activity
acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1]") #carboxylic acids only
seen = set()

def get_pkas(cid):
    #multiple tries in case of chembl timeout
    for tries in range(5):
        try:
            vals = []
            for a in acts.filter(molecule_chembl_id=cid, standard_type="pKa"):
                try:
                    vals.append(float(a["standard_value"]))
                except:
                    pass
            return vals
        except RequestException:
            time.sleep(2 ** tries)
    return []

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["chembl_id", "canonical_smiles", "pkas", "mean_pka"])

    #common carboxylic acid SMILES fragments
    for query in ["C(=O)O", "OC(=O)", "O=C(O)"]:
        for r in mols.filter(molecule_structures__canonical_smiles__contains=query):
            cid = r["molecule_chembl_id"]
            if cid in seen:
                continue
            seen.add(cid)

            smi = (r.get("molecule_structures") or {}).get("canonical_smiles")
            mol = Chem.MolFromSmiles(smi) if smi else None

            #confirm COOH group with SMARTS 
            if not mol or not mol.HasSubstructMatch(acid):
                continue

            pkas = get_pkas(cid)
            if pkas:
                w.writerow([cid, smi, ";".join(map(str, pkas)), mean(pkas)])

print("saved", OUT)