import pandas as pd
from ast import literal_eval
from rdkit import DataStructs
import numpy as np
from ast import literal_eval
from scipy import stats
from scipy.stats import false_discovery_control


def main():
    df_fp = pd.read_pickle("../../results/schembl_summs_v5_final_fp_bitvec.pkl")
    df_fp["summarizations"] = df_fp["summarizations"].apply(literal_eval)

    # load all labels
    with open("../../results/all_labels.txt", "r") as f:
        all_labels = f.read().splitlines()

    t_stats = []
    p_vals = []
    n_mols = []

    # for term in all_labels[-2:]:
    for count, term in enumerate(all_labels):
        print(f"{count}/{len(all_labels)} -- {term}", end="\r")
        df_term = df_fp[df_fp["summarizations"].map(lambda x: any(term == word for word in x))].reset_index(drop=True)
        df_rand = df_fp.sample(n=len(df_term), random_state=42).reset_index(drop=True)
        fp_tanimoto = np.zeros(len(df_term))
        fp_tanimoto_rand = np.zeros(len(df_term))
        for i in range(len(df_term)):
            # calculate the max bulk tanimoto similarity between the query and the other molecules
            fp_tanimoto[i] = max(DataStructs.BulkTanimotoSimilarity(df_term["fingerprint"].iloc[i], df_term.drop(i, axis=0).reset_index(drop=True)["fingerprint"]))
            fp_tanimoto_rand[i] = max(DataStructs.BulkTanimotoSimilarity(df_rand["fingerprint"].iloc[i], df_rand.drop(i, axis=0).reset_index(drop=True)["fingerprint"]))

        # compute t test p value
        t_stat, p_val = stats.ttest_ind(fp_tanimoto, fp_tanimoto_rand)
        t_stats.append(t_stat)
        p_vals.append(p_val)
        n_mols.append(len(df_term))

    # correct for multiple testing with FDR
    p_vals_corrected = false_discovery_control(p_vals)


    df_p_values = pd.DataFrame({"label": all_labels, "t_stat": t_stats, "p_val": p_vals, "p_val_corrected": p_vals_corrected, "n_mols": n_mols})

    df_p_values.to_csv("primary_label_t_test.csv", index=False)


if __name__ == "__main__":
    main()