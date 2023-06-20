import time
import requests
import argparse
import pandas as pd
import numpy as np
import multiprocessing as mp
import openai
from pathlib import Path

from itertools import repeat
from rdkit import Chem
# from pandarallel import pandarallel
from bs4 import BeautifulSoup


from functional_similarity import chatgpt_functional_similarity
from patent_summarization import load_data, get_cids, get_patents_from_cids, list_to_set, patent_coverage, get_patent_info, chatgpt_summarization, summarization_wrapper, summarizations_to_str



def main(args):
    i_paths = [
        # "chess_v1_top_results/pen_oechem_1k.csv",
        # "chess_v1_top_results/pen_rdkit_atom_0_1k.csv",
        # "chess_v1_top_results/pen_rdkit_atom_n_1k.csv",
        # "chess_v1_top_results/pax_oechem_1k.csv",
        # "chess_v1_top_results/pax_rdkit_atom_0_1k.csv",
        # "chess_v1_top_results/pax_rdkit_atom_n_1k.csv",
        # "chess_v1_top_results/azt_oechem_1k.csv",
        # "chess_v1_top_results/azt_rdkit_atom_0_1k.csv",
        # "chess_v1_top_results/azt_rdkit_atom_n_1k.csv",
        # "chess_v1_top_results/lsd_oechem_1k.csv",
        # "chess_v1_top_results/lsd_rdkit_atom_0_1k.csv",
        # "chess_v1_top_results/lsd_rdkit_atom_n_1k.csv",
        # "chess_v1_top_results/fentanyl_oechem_1k.csv",
        # "chess_v1_top_results/fentanyl_rdkit_atom_0_1k.csv",
        # "chess_v1_top_results/fentanyl_rdkit_atom_n_1k.csv",
        # "chess_v1_top_results/avobenzone_oechem_1k.csv",
        # "chess_v1_top_results/avobenzone_rdkit_atom_0_1k.csv",
        # "chess_v1_top_results/avobenzone_rdkit_atom_n_1k.csv",
        # "chess_v1_top_results/ab25_oechem_1k.csv",
        # "chess_v1_top_results/ab25_rdkit_atom_0_1k.csv",
        # "chess_v1_top_results/ab25_rdkit_atom_n_1k.csv",
        # "chess_v1_top_results/2-dpac_oechem_1k.csv",
        # "chess_v1_top_results/2-dpac_rdkit_atom_0_1k.csv",
        # "chess_v1_top_results/2-dpac_rdkit_atom_n_1k.csv",
        # "chess_v1_top_results/rsv_inhibitor.csv",
        # "chess_v1_top_results/rsv_inhibitor_canonicalized.csv",
        "chess_v1_top_results/morphine.csv",
    ]

    q_paths = [
        # "query_descriptors/pen.csv",
        # "query_descriptors/pen.csv",
        # "query_descriptors/pen.csv",
        # "query_descriptors/pax.csv",
        # "query_descriptors/pax.csv",
        # "query_descriptors/pax.csv",
        # "query_descriptors/azt.csv",
        # "query_descriptors/azt.csv",
        # "query_descriptors/azt.csv",
        # "query_descriptors/lsd.csv",
        # "query_descriptors/lsd.csv",
        # "query_descriptors/lsd.csv",
        "query_descriptors/fentanyl.csv",
        # "query_descriptors/fentanyl.csv",
        # "query_descriptors/fentanyl.csv",
        # "query_descriptors/avobenzone.csv",
        # "query_descriptors/avobenzone.csv",
        # "query_descriptors/avobenzone.csv",
        # "query_descriptors/ab25.csv",
        # "query_descriptors/ab25.csv",
        # "query_descriptors/ab25.csv",
        # "query_descriptors/2-dpac.csv",
        # "query_descriptors/2-dpac.csv",
        # "query_descriptors/2-dpac.csv",
        # "query_descriptors/pax.csv",
        # "query_descriptors/pax.csv",
    ]
    output_dir = "chess_v1_functional_similarity"
    # output_dir = "chess_v2_zinc_functional_similarity"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    input_type = "csv"
    head_n = 100
    desc_len = 3500
    gpt_max_patents_per_mol = 10
    gpt_temperature = 0
    gpt_model = "gpt-3.5-turbo"
    api_key = "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"
    gpt_summ_system_prompt = "You are an organic chemist summarizing chemical patents"
    gpt_summ_user_prompt = "Output a brief list of 1-3 word phrases that best describe the chemical and pharmacological function(s) of the molecule described by the given patent abstract and description. Be specific, concise, and follow the syntax 'descriptor_1 / descriptor_2 / etc', writing 'NA' if nothing is provided. The following is the patent: "
    gpt_func_system_prompt = "You are a chemistry expert comparing functional similarity of chemicals"
    # gpt_func_user_prompt = "Pretend you are a chemistry expert. Given sets of functional descriptors for two molecules, determine if the two molecules have similar specific chemical function. Respond in the format: '{25 word maximum reasoning} --- {Yes or No}'"
    # gpt_func_user_prompt = "Given functional descriptors of two molecules, determine if the two molecules have similar function. If similar functions are found in both lists, these molecules have similar function. Respond in the format: '{yes or no} --- {20 word maximum explanation}'"
    openai.api_key = api_key

    # n_cpus = mp.cpu_count()
    n_cpus = 6
    print(f"INFO: Using {n_cpus} CPUs")

    for i_path, q_path in zip(i_paths, q_paths):
        start = time.time()
        query_df = pd.read_csv(q_path)
        print(i_path)
        output_path = f"{output_dir}/{i_path.split('/')[-1]}"
        df = load_data(i_path, input_type, head_n)

        with mp.Pool(n_cpus) as p:
            print("INFO: Getting cids...")
            cids = p.map(get_cids, df['smiles'].tolist())
            print("INFO: Getting patent ids...")
            patent_ids = p.map(get_patents_from_cids, cids)
            patent_ids = p.map(list_to_set, patent_ids)
            coverage = p.map(patent_coverage, patent_ids)
            print("INFO: Getting patent info...")
            patent_info = p.starmap(get_patent_info, zip(patent_ids, repeat(gpt_max_patents_per_mol)))
            print(f"INFO: Summarizing patents with {gpt_model}...")
            summarizations_sources = p.starmap(summarization_wrapper, zip(patent_info, repeat(api_key), repeat(gpt_model), repeat(gpt_summ_system_prompt), repeat(gpt_summ_user_prompt), repeat(desc_len), repeat(gpt_temperature), repeat(gpt_max_patents_per_mol)))
            summarizations = p.map(summarizations_to_str, summarizations_sources)
            print(f"INFO: Determining functional similarity with {gpt_model}...")
            functional_similarity = p.starmap(chatgpt_functional_similarity, zip(repeat(api_key), repeat(gpt_model), repeat(gpt_func_system_prompt), repeat(""), repeat(str(query_df["summarizations"][0])), summarizations, repeat(gpt_temperature)))

        df['cids'] = [c for c in cids]
        df['patent_ids'] = [p for p in patent_ids]
        df['coverage'] = coverage
        df['patent_info'] = [p for p in patent_info]
        df['summarizations_sources'] = [s for s in summarizations_sources]
        df['summarizations'] = [s for s in summarizations]
        df['functional_similarity'] = [f for f in functional_similarity]
        df[["smiles", "cids", "coverage", "summarizations_sources", "summarizations", "functional_similarity"]].to_csv(output_path, index=False)
        end = time.time()
        print(f"Time elapsed: {end - start} seconds")
        print("Success!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)