import time
import pandas as pd

from pandarallel import pandarallel
from rdkit import Chem
from collections import defaultdict


pandarallel.initialize(progress_bar=False)

def mol_to_inchi_key(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(mol)
    except:
        return None


def main():
    t0 = time.time()
    df = pd.read_csv('../data/surechembl_smiles_canon_chiral_randomized.csv')
    df = df.rename(columns={"SMILES": "smiles"})

    print("SMILES to InChI Key")
    df['inchi_key'] = df['smiles'].parallel_apply(mol_to_inchi_key)
    df = df[df['inchi_key'] != None].reset_index(drop=True)
    print(f"Time Elapsed: {time.time() - t0} seconds\n")
    
    print("Loading InChI Key to CID Dictionary")
    inchi_to_cid = {}
    with open('../data/CID-InChI-Key') as f:
        for line in f:
            (val, _, key) = line.split()
            inchi_to_cid[key] = val
    print(f"Time Elapsed: {time.time() - t0} seconds\n")

    print("InChI Key to CID")
    df['cid'] = df['inchi_key'].parallel_apply(lambda x: inchi_to_cid[x] if x in inchi_to_cid else None)
    df = df[df['cid'] != None].reset_index(drop=True)
    print(f"Time Elapsed: {time.time() - t0} seconds\n")

    print("Loading CID to Patent Dictionary")
    cid_to_patents = defaultdict(set)
    with open('../data/CID-Patent') as f:
        for line in f:
            (cid, patent) = line.split()
            cid_to_patents[cid].update(patent) # there are multiple patents per cid
    print(f"Time Elapsed: {time.time() - t0} seconds\n")

    print("CID to Patent")
    df['patent_ids'] = df['cid'].parallel_apply(lambda x: cid_to_patents[x] if x in cid_to_patents else None)
    df = df[df['patent_ids'] != None].reset_index(drop=True)
    print(f"Time Elapsed: {time.time() - t0} seconds\n")

    print("Saving")
    df["smiles", "cid", "patent_ids"].to_pickle('../data/surechembl_smiles_canon_chiral_randomized_patents.pkl')
    df["smiles", "cid", "patent_ids"].to_csv('../data/surechembl_smiles_canon_chiral_randomized_patents.csv', index=False)
    print(f"Time Elapsed: {time.time() - t0} seconds\n")




if __name__ == "__main__":
    main()