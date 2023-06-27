import time
import pandas as pd

from pandarallel import pandarallel
from rdkit import Chem
from collections import defaultdict


pandarallel.initialize(progress_bar=False)

# NOTE Need to update to remove pandarallel and instead map on index. Wayyyy faster that way.

def mol_to_inchi_key(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(mol)
    except:
        return None


def main():
    t_curr = time.time()
    df = pd.read_csv('../data/surechembl_smiles_canon_chiral_randomized.csv')
    df = df.rename(columns={"SMILES": "smiles"})
    print(f"Loaded smiles into dataframe. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    df['inchi_key'] = df['smiles'].parallel_apply(mol_to_inchi_key)
    df = df[df['inchi_key'] != None].reset_index(drop=True)
    print(f"smiles mapped to inchi_key. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")
    
    inchi_to_cid = {}
    with open('../data/CID-InChI-Key') as f:
        for line in f:
            (val, _, key) = line.split()
            inchi_to_cid[key] = val
    print(f"Loaded CID-InChI-key dictionary. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    df['cid'] = df['inchi_key'].parallel_apply(lambda x: inchi_to_cid[x] if x in inchi_to_cid else None)
    df = df[df['cid'] != None].dropna().reset_index(drop=True)
    print(f"Mapped cid to inchi_key. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    df[["smiles", "cid"]].to_csv('../data/surechembl_smiles_canon_chiral_randomized_cids.csv', index=False)
    print(f"Saved as csv. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")




if __name__ == "__main__":
    main()