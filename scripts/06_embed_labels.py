
import pandas as pd
import openai
import time
import multiprocessing as mp
from itertools import repeat



def openai_embed_text(text, api_key, model = "text-embedding-ada-002"):
    """
    
    Parameters:
    -----------

    Returns:
    --------
    """
    api_success = False
    tries = 0
    while api_success == False:
        try:
            openai.api_key = api_key
            response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']
        except Exception as e:
            tries += 1
            time.sleep(5)
            if tries > 10:
                print(e)
                return None


def main():
    api_key = "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"
    with open("../results/schembl_summarizations_vocab_v3_alg_cleaned.txt") as f:
        labels = f.read().splitlines()

    df = pd.DataFrame(labels, columns = ["labels"])

    n_cpus = 16
    with mp.Pool(n_cpus) as p:
        embeddings = p.starmap(openai_embed_text, zip(labels, repeat(api_key)))

    df['embeddings'] = embeddings

    df.to_pickle("../results/schembl_summarizations_vocab_v4_embeddings.pkl")
    df.to_csv("../results/schembl_summarizations_vocab_v4_embeddings.csv")

if __name__ == "__main__":
    main()