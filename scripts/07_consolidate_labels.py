
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
from ast import literal_eval
from pandarallel import pandarallel




def main():
    pandarallel.initialize(progress_bar = True)
    # df = pd.read_pickle("../results/schembl_summs_v3_vocab_embeddings.pkl")
    df = pd.read_csv("../results/schembl_summs_v3_vocab_embeddings.csv")
    print('loaded df')
    df['embeddings'] = df['embeddings'].parallel_apply(literal_eval)
    df['labels'] = df['labels'].parallel_apply(str)
    df.to_pickle("../results/schembl_summs_v3_vocab_embeddings.pkl")
    print('literal eval done')


    arr = np.array(df['embeddings'].tolist())
    print('array made')

    eps_diff_dir = Path("../results/eps_diffs")
    eps_diff_dir.mkdir(parents = True, exist_ok = True)
    # perform dbscan clustering
    prev_set = set()

    for eps in range(5, 100, 5):
    # for eps in range(30, 35, 1):
    # for eps in range(340, 341, 1):
        if eps == 0:
            continue
        eps_ = eps / 1000
        print(f"eps {eps_}")
        dbscan = DBSCAN(eps = eps_, min_samples = 1,n_jobs=-1)
        dbscan.fit(arr)
        df['dbscan'] = dbscan.labels_
        df_grouped = df[['labels', 'dbscan']]

        # convert each cluster to a set, then add to a set of sets
        curr_set = set()
        for cluster in df_grouped['dbscan'].unique():
            curr_set.add(frozenset(df_grouped[df_grouped['dbscan'] == cluster]['labels'].tolist()))
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


    # NOTE Choosing eps of 34 because 35 resulted in a large kinase / non-kinase cluster that would be inaccurate.
    # NOTE From what I can tell, eps 34 has no major offenders

if __name__ == "__main__": 
    main()