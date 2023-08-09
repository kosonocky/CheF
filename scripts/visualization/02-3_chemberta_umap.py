from umap import UMAP
import pandas as pd
from rdkit import Chem
import numpy as np
from ast import literal_eval



def main():

    df = pd.read_pickle("schembl_summs_v4_final_with_chemberta.pkl")
    df = df.dropna(subset=["features"])
    print("Dataframe loaded")

    # create tsne of fingerprint data
    model = UMAP(n_components=2, verbose=1, metric="euclidean",n_jobs=44)
    umap_results = model.fit_transform(np.array(df["features"].tolist()))

    # add tsne results to dataframe
    df["cb_umap_x"] = umap_results[:,0]
    df["cb_umap_y"] = umap_results[:,1]

    # save dataframe
    df.to_pickle("schembl_summs_v4_final_with_chemberta_umap.pkl")
    df.to_csv("schembl_summs_v4_final_with_chemberta_umap.csv", index=False)



if __name__ == "__main__":
    main()