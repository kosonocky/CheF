import pandas as pd
from rdkit import Chem
import os


def main():

    with open("schembl_summs_v4_gpt_cleaned_eps_0.340_diff_20030_final_canon.csv", 'r') as f:
        df = pd.read_csv(f)

    # # canonicalize smiles
    # df['canon_smiles'] = df['smiles'].apply(lambda x: Chem.CanonSmiles(x, useChiral=False))

    # df.to_csv("schembl_summs_v4_gpt_cleaned_eps_0.340_diff_20030_final_canon.csv")

    # drop duplicates on canon_smiles
    df = df.drop_duplicates(subset=['canon_smiles'])
    # create features col of empty lists
    df["features"] = df["canon_smiles"].apply(lambda x: [])
    
    # set canon_smiles as index
    df = df.set_index('canon_smiles')


    features_df = pd.DataFrame()

    # open each file in path
    path = "../../../featurized_smiles/ChemBERTa/surechembl"
    for filename in os.listdir(path):
        if filename.endswith(".pkl"):
            print(filename)
            pkl_df = pd.read_pickle(os.path.join(path, filename)).rename(columns={"SMILES":"canon_smiles"})
            # convert pickle df to dictionary converting canon_smiles to features
            # pkl_dict = pkl_df.set_index('canon_smiles').T.to_dict('list')
            # map features to df, defaulting to existing value if not found
            # df["features"] = df.index.map(lambda x: pkl_dict.get(x, df.loc[df.index == x, 'features'].iloc[0]))
            # df["features"] = df["canon_smiles"].map(lambda x: pkl_dict.get(x, df.loc[df['canon_smiles'] == x, 'features'].iloc[0]))            
            # df.to_csv("test.csv", index=False)


            # add pkl_df to features_df only if the smiles are found in df
            features_df = pd.concat([features_df, pkl_df[pkl_df['canon_smiles'].isin(df.index)]])
    
    # merge features_df with df
    df = df.merge(features_df, on='canon_smiles', how='left')

    df.to_pickle("schembl_summs_v4_gpt_cleaned_eps_0.340_diff_20030_final_chemberta.pkl")
    df.to_csv("schembl_summs_v4_gpt_cleaned_eps_0.340_diff_20030_final_chemberta.csv")

    pass



if __name__ == '__main__':
    main()