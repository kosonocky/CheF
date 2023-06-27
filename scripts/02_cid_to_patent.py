import time
import pandas as pd
import pickle as pkl

# from pandarallel import pandarallel
from collections import defaultdict


# pandarallel.initialize(progress_bar=False)



def main():
    t_curr = time.time()

    # NOTE This section of code creates the CID-Patent ID Dictionary. It is commented out because it is saved/loaded as pkl in future
    # print("Loading CID to Patent Dictionary")
    # cid_to_patents = defaultdict(set)
    # with open('../data/CID-Patent') as f:
    #     for line in f:
    #         (cid, patent) = line.split()
    #         cid_to_patents[int(cid)].update([patent]) # there are multiple patents per cid
    # with open("cid_patent_dict.pkl", 'wb') as f:
    #     pkl.dump(cid_to_patents, f)
    # print(f"Loaded dictionary and saved as pkl. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")
    # print("Saved to pkl")

    # NOTE This section of code takes the CID-Patent ID Dictionary and removes all entries > 10 patents
    # # This is done to ignore over-patented molecules that will results in less accurate functional descriptors
    # # remove all k,v pairs in cid_to_patents that have more than 10 patents in value set
    # cid_to_patents_l10p = {k: v for k, v in cid_to_patents.items() if len(v) <= 10}
    # print("Removed all k,v pairs in cid_to_patents that have more than 10 patents in value set")
    # with open("cid_patent_dict_l10p.pkl", 'wb') as f:
    #     pkl.dump(cid_to_patents_l10p, f)
    # print(f"Removed entries w/ more than 10 patents. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    # Open the CID-Patent ID Dictionary with only 10 or less patents per CID
    with open("cid_patent_dict_l10p.pkl", "rb") as f:
        cid_to_patents = pkl.load(f)
    print("Loaded pkl")
    print(f"Loaded dict pkl. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    # Load the CID-SMILES DataFrame
    df = pd.read_csv('../data/surechembl_smiles_canon_chiral_randomized_cids.csv')
    df = df.dropna().reset_index(drop=True) # drop all rows with NaN
    df["cid"] = df["cid"].map(int) # convert cid to int
    df = df.set_index("cid") # make cid col the index for faster mapping. Make sure not to reset index after this point or else cid will be lost
    print(f"Loaded CID-SMILES dataframe. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    # Map CID to Patent ID dictionary to df
    df['patent_ids'] = df.index.map(lambda x: cid_to_patents.get(x))
    df = df[df['patent_ids'] != None].dropna() # do not reset index, they represent cid
    print(f"Patents mapped to CID. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")

    df[["smiles", "patent_ids"]].to_csv(f'../data/surechembl_smiles_canon_chiral_randomized_patents.csv', index=True)
    print(f"Saved as csv. Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")




if __name__ == "__main__":
    main()