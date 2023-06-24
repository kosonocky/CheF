import time
import pandas as pd

from pandarallel import pandarallel
from collections import defaultdict


pandarallel.initialize(progress_bar=False)



def main():
    t0 = time.time()
    df = pd.read_csv('../data/surechembl_smiles_canon_chiral_randomized_cids.csv')

    print("Loading CID to Patent Dictionary")
    cid_to_patents = defaultdict(set)
    with open('../data/CID-Patent') as f:
        for line in f:
            (cid, patent) = line.split()
            cid_to_patents[cid].update(patent) # there are multiple patents per cid
    print(f"Time Elapsed: {time.time() - t0} seconds\n")

    # split into 10 chunks
    for i in range(0,len(df),len(df)//10):
        print(f"Processing {i} to {i+len(df)//10}")
        if len(df) - i < len(df)//10:
            tmp_df = df[i:]
        else:
            tmp_df = df[i:i+len(df)//10]
        
        print("CID to Patent")
        tmp_df['patent_ids'] = tmp_df['cid'].parallel_apply(lambda x: cid_to_patents[x] if x in cid_to_patents else None)
        tmp_df = tmp_df[tmp_df['patent_ids'] != None].reset_index(drop=True)
        print(f"Time Elapsed: {time.time() - t0} seconds\n")

        print("Saving")
        tmp_df[["smiles", "cid", "patent_ids"]].to_pickle(f'../data/surechembl_smiles_canon_chiral_randomized_patents_{i}.pkl')
        tmp_df[["smiles", "cid", "patent_ids"]].to_csv(f'../data/surechembl_smiles_canon_chiral_randomized_patents_{i}.csv', index=False)
        print(f"Time Elapsed: {time.time() - t0} seconds\n")




if __name__ == "__main__":
    main()