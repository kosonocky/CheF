import time
import pandas as pd

from rdkit import Chem
from collections import defaultdict

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

    df = df.set_index("smiles")
    df['inchi_key'] = df.index.map(lambda x: mol_to_inchi_key(x))
    df["smiles"] = df.index

    df = df[df['inchi_key'] != None].reset_index(drop=True)
    print(f"smiles mapped to inchi_key. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")
    
    inchi_to_cid = {}
    with open('../data/CID-InChI-Key') as f:
        for line in f:
            (val, _, key) = line.split()
            inchi_to_cid[key] = val
    print(f"Loaded CID-InChI-key dictionary. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")


    df = df.set_index("inchi_key")
    df['cid'] = df.index.map(lambda x: inchi_to_cid[x] if x in inchi_to_cid else None)

    df = df[df['cid'] != None].dropna().reset_index(drop=True)
    print(f"Mapped cid to inchi_key. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    df[["smiles", "cid"]].to_csv('../data/surechembl_smiles_canon_chiral_randomized_cids.csv', index=False)
    print(f"Saved as csv. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")




if __name__ == "__main__":
    main()