from sklearn.manifold import TSNE
import pandas as pd
from rdkit import Chem
import numpy as np
from ast import literal_eval



def main():

    df = pd.read_pickle("schembl_summs_v4_final_with_chemberta.pkl")
    df = df.dropna(subset=["features"])
    print("Dataframe loaded")

    # create tsne of fingerprint data
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, metric="euclidean",n_jobs=44)
    tsne_results = tsne.fit_transform(np.array(df["features"].tolist()))

    # add tsne results to dataframe
    df["chemberta_tsne_x"] = tsne_results[:,0]
    df["chemberta_tsne_y"] = tsne_results[:,1]

    # save dataframe
    df.to_pickle("schembl_summs_v4_final_with_chemberta_tsne.pkl")
    df.to_csv("schembl_summs_v4_final_with_chemberta_tsne.csv", index=False)



if __name__ == "__main__":
    main()