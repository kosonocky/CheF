import pandas as pd
from ast import literal_eval
from rdkit import DataStructs
import numpy as np
from ast import literal_eval
from scipy import stats
from collections import Counter


def main():
    df = pd.read_csv("gephi_graphs/graph_data.csv")
    counter = Counter()
    df_fp = pd.read_pickle("../../results/schembl_summs_v5_final_fp_bitvec.pkl")
    df_fp["summarizations"] = df_fp['summarizations'].apply(literal_eval)
    # count each label in summarizations
    for i in range(len(df_fp)):
        counter.update(df_fp["summarizations"].iloc[i])

    # load all labels
    with open("../../results/all_labels.txt", "r") as f:
        all_labels = f.read().splitlines()

    t_stats = []
    p_vals = []
    n_mols = []

    for count, term_1 in enumerate(all_labels):
        print(f"{count}/{len(all_labels)} -- {term_1}", end="\r")
        # swap source and target if term_1 is in target. This ensures that term_1 is always in source
        df.loc[df["target"] == term_1, ["target", "source"]] = df.loc[df["target"] == term_1, ["source", "target"]].values
        df["source_global_count"] = df['source'].apply(lambda x: counter[x])
        df["target_global_count"] = df['target'].apply(lambda x: counter[x])

        # get 10 closest terms to term_1 that aren't very common
        close_terms = df.loc[df["source"] == term_1].loc[df["target_global_count"] < 1000].sort_values(by="weight", ascending=False).head(10)["target"].values

        # for each row in dataframe that contains term_1 in 'summarizations'
        df_term_1 = df_fp[df_fp["summarizations"].map(lambda x: any(term_1 == word for word in x))].reset_index(drop=True)

        # for each row in dataframe that contains any term in close_terms in 'summarizations'
        df_close_terms = df_fp[df_fp["summarizations"].map(lambda x: any(term in x for term in close_terms))].reset_index(drop=True)

        # Compute mean distance from term_1 to the closest term in term_2
        fp_tanimoto = np.zeros((len(df_term_1)))
        for i in range(len(df_term_1)):
            sim = np.array(DataStructs.BulkTanimotoSimilarity(df_term_1['fingerprint'].iloc[i], df_close_terms['fingerprint']))
            fp_tanimoto[i] = np.max(sim)

        # null similarity. select random 10 labels each time
        fp_tanimoto_rand = np.zeros((len(df_term_1)))
        df_rand = df_fp.sample(n=len(df_close_terms), random_state=42).reset_index(drop=True)
        for i in range(len(df_term_1)):
            # rand_10_terms = random.sample([term for term in counter if term not in illegal_terms and counter[term] < 1000], 10)
            # df_random_terms = df_fp[df_fp["summarizations"].map(lambda x: any(term in x for term in rand_10_terms))].reset_index(drop=True)
            sim = np.array(DataStructs.BulkTanimotoSimilarity(df_term_1['fingerprint'].iloc[i], df_rand['fingerprint']))
            fp_tanimoto_rand[i] = np.max(sim)

        # compute t test p value
        t_stat, p_val = stats.ttest_ind(fp_tanimoto, fp_tanimoto_rand)
        t_stats.append(t_stat)
        p_vals.append(p_val)
        n_mols.append(len(df_term_1))

    df_p_values = pd.DataFrame({"label": all_labels, "t_stat": t_stats, "p_val": p_vals, "n_mols": n_mols})

    df_p_values.to_csv("co-occurrence_t_test.csv", index=False)


if __name__ == "__main__":
    main()