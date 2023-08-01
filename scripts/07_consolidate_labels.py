
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from pathlib import Path





def main():
    df = pd.read_pickle("../results/surechembl_smiles_canon_chiral_randomized_patents_l10p_summarizations/surechembl_summarizations_cleaned_vocab_embeddings.pkl")
    arr = np.array(df['embeddings'].tolist())


    eps_diff_dir = Path("../results/surechembl_smiles_canon_chiral_randomized_patents_l10p_summarizations/eps_diffs")
    eps_diff_dir.mkdir(parents = True, exist_ok = True)
    # perform dbscan clustering
    prev_set = set()

    for eps in range(0, 100, 5):
        if eps == 0:
            continue
        eps_ = eps / 100
        print(f"eps {eps_}")
        dbscan = DBSCAN(eps = eps_, min_samples = 1,)
        dbscan.fit(arr)
        df['dbscan'] = dbscan.labels_
        df = df[['label', 'dbscan']]
        df_grouped = df

        # convert each cluster to a set, then add to a set of sets
        curr_set = set()
        for cluster in df_grouped['dbscan'].unique():
            curr_set.add(frozenset(df_grouped[df_grouped['dbscan'] == cluster]['label'].tolist()))    
        # convert frozen set to single string of alphabetized entries
        curr_set = [sorted(list(x)) for x in curr_set]
        curr_set = [", ".join(x) for x in curr_set]
        curr_set = set(curr_set)

        with open(eps_diff_dir / f"eps_{str(eps).zfill(3)}_diff_{len(set(dbscan.labels_))}_clusters.txt", "w") as f:
            for item in sorted(curr_set):
                f.write(f"{item}\n")

        # compare current set of sets to previous set of sets
        if curr_set != prev_set:
            # save difference in sets
            diff = curr_set - prev_set
            with open(eps_diff_dir / f"eps_{str(eps).zfill(3)}_diff_{len(set(dbscan.labels_))}_diffs.txt", "w") as f:
                for item in diff:
                    f.write(f"{item}\n")
            # update prev set
            prev_set = curr_set


    # # compute cosine sim
    # cos_sim = cosine_similarity(arr, arr)
    # # save in readable csv format, with labels as columns and index
    # df_cos_sim = pd.DataFrame(cos_sim, columns = df['label'].tolist(), index = df['label'].tolist())
    # df_cos_sim.to_csv("../results/surechembl_smiles_canon_chiral_randomized_patents_l10p_summarizations/surechembl_summarizations_top-100_gpt-3.5-turbo_desc-3500_cleaned_terms_cos_sim.csv")


if __name__ == "__main__": 
    main()