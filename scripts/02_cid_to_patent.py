import time
import pandas as pd
import pickle as pkl

# from pandarallel import pandarallel
from collections import defaultdict


# pandarallel.initialize(progress_bar=False)



def main():
    t0 = time.time()

    # NOTE This section of code creates the CID-Patent ID Dictionary. It is commented out because it is saved/loaded as pkl in future
    # print("Loading CID to Patent Dictionary")
    # cid_to_patents = defaultdict(set)
    # with open('../data/CID-Patent') as f:
    #     for line in f:
    #         (cid, patent) = line.split()
    #         cid_to_patents[int(cid)].update([patent]) # there are multiple patents per cid
    # with open("cid_patent_dict.pkl", 'wb') as f:
    #     pkl.dump(cid_to_patents, f)
    # print("Saved to pkl")
    # print(f"Time Elapsed: {time.time() - t0} seconds\n")

    # NOTE This section of code takes the CID-Patent ID Dictionary and removes all entries > 10 patents
    # # remove all k,v pairs in cid_to_patents that have more than 10 patents in value set
    # cid_to_patents_l10p = {k: v for k, v in cid_to_patents.items() if len(v) <= 10}
    # print("Removed all k,v pairs in cid_to_patents that have more than 10 patents in value set")
    # with open("cid_patent_dict_l10p.pkl", 'wb') as f:
    #     pkl.dump(cid_to_patents_l10p, f)
    # print(f"Time Elapsed: {time.time() - t0} seconds\n")

    with open("cid_patent_dict.pkl", "rb") as f:
        cid_to_patents = pkl.load(f)
    print("Loaded pkl")
    print(f"Time Elapsed: {time.time() - t0} seconds\n")

    df = pd.read_csv('../data/surechembl_smiles_canon_chiral_randomized_cids.csv')
    df = df.dropna().reset_index(drop=True)
    df["cid"] = df["cid"].map(int)
    # make cid index for faster mapping later on
    df = df.set_index("cid")

    print("df loaded")
    print(f"Time Elapsed: {time.time() - t0} seconds\n")

    df['patent_ids'] = df.index.map(lambda x: cid_to_patents.get(x))
    df = df[df['patent_ids'] != None] # do not reset index, they represent cid
    df = df.dropna()
    print("Patents mapped to CID")
    print(f"Time Elapsed: {time.time() - t0} seconds\n")

    print("Saving")
    df[["smiles", "patent_ids"]].to_pickle(f'../data/surechembl_smiles_canon_chiral_randomized_patents.pkl')
    df[["smiles", "patent_ids"]].to_csv(f'../data/surechembl_smiles_canon_chiral_randomized_patents.csv', index=True)
    print(f"Time Elapsed: {time.time() - t0} seconds\n")




if __name__ == "__main__":
    main()