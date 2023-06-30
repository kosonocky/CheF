
import pandas as pd
import openai
import time



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
    with open("../results/surechembl_smiles_canon_chiral_randomized_patents_l10p_summarizations/surechembl_summarizations_cleaned_vocab.txt") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    df = pd.DataFrame(lines, columns = ["label"])
    df['embeddings'] = df['label'].apply(lambda x: openai_embed_text(x, api_key))
    df.to_pickle("../results/surechembl_smiles_canon_chiral_randomized_patents_l10p_summarizations/surechembl_summarizations_cleaned_vocab_embeddings.pkl")
    df.to_csv("../results/surechembl_smiles_canon_chiral_randomized_patents_l10p_summarizations/surechembl_summarizations_cleaned_vocab_embeddings.csv")

if __name__ == "__main__":
    main()