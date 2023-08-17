import pandas as pd
from ast import literal_eval
from collections import Counter
import pickle as pkl


def main():
    df = pd.read_csv("../../results/schembl_summs_v5_final.csv")
    df["summarizations"] = df["summarizations"].apply(literal_eval)

    with open("../../results/all_labels.txt", "r") as f:
        all_labels = f.read().splitlines()

    counter = Counter()
    for label_1 in all_labels:
        for label_2 in all_labels:
            if label_1 != label_2:
                for summarization in df["summarizations"]:
                    if label_1 in summarization and label_2 in summarization:
                        counter[(label_1, label_2)] += 1
                        
    # save counter
    with open("../../results/counter.pkl", "wb") as f:
        pkl.dump(counter, f)

    # save counter as csv
    df = pd.DataFrame.from_dict(counter, orient="index")
    df.to_csv("../../results/counter.csv")

if __name__ == '__main__':
    main()