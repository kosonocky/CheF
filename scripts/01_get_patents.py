import time
import requests
import argparse
import pandas as pd
import multiprocessing as mp
import openai
from bs4 import BeautifulSoup
import sys
from pathlib import Path
from itertools import repeat


def load_data(input_:str, input_type:str = "csv", head_n:int = 0) -> pd.DataFrame:
    """
    Loads data from a csv file or single SMILES string into a pandas dataframe.

    Parameters:
    -----------
    input_ : str
        Path to the input file or SMILES string.
    input_type : str
        Type of input. Either "csv" or "smiles".
    head_n : int
        Number of rows to read from the csv file. If 0, reads entire file.

    Returns:
    --------
    df : pandas dataframe
        Dataframe containing SMILES strings.

    """

    if input_type == "csv":
        df = pd.read_csv(input_)
        if head_n != 0: # option to select entire file
            if head_n < len(df):
                df = df[:head_n]
        df = df.rename(columns={"SMILES": "smiles"})
    else:
        df = pd.DataFrame({"smiles": [input_]})
    return df


def pub_api_call(url, params = None, _n_restarts = 0):
    """
    Calls the PubChem API. If the API is overloaded, it will wait and try again.

    Parameters:
    -----------
    url : str
        URL for the PubChem API.
    params : dict
        Parameters for the PubChem API.
    _n_restarts : int
        Number of times the PubChem API has restarted. Do not specify when calling this function.

    Returns:
    --------
    response : requests response
        Response from the PubChem API.
    """

    if params is not None:
        response = requests.get(url, params = params)
    else:
        response = requests.get(url)
    if response.status_code == 503: # 503 returned when PubChem API is overloaded
        if _n_restarts <= 5:
            if _n_restarts > 3:
                print(f"WARNING: PubChem API Failed and will restart - n_restarts: {_n_restarts}")
                sys.stdout.flush()
            t = response.headers['Retry-After']
            time.sleep(int(t))
            return pub_api_call(url, params, _n_restarts = _n_restarts + 1)
    return response



def get_cids(smiles, _neutralize_charges = False, id_match_type = "same_connectivity"):
    """
    Gets PubChem CIDs for all stereoisomers of a SMILES string.

    If it cannot find it with the given string, it will attempt to neutralize all charges according to PubChem's rules and try again.
    
    Parameters:
    -----------
    smiles : str
        SMILES string.
    _neutralize_charges : bool
        Whether to attempt to neutralize charges and try again.
    
    Returns:
    --------
    cids : list
        List of PubChem CIDs.
    """

    params = {'smiles': smiles}
    if id_match_type == "same_connectivity":
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastidentity/smiles/cids/txt?identity_type=same_connectivity"
    elif id_match_type == "exact_match":
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastidentity/smiles/cids/txt"
    else:
        raise ValueError(f"id_match_type must be either 'same_connectivity' or 'exact_match', not {id_match_type}")
    
    response = pub_api_call(url = url, params = params)
    if response.ok:
        cids = [int(cid) for cid in response.text.split('\n') if cid != '']
        return cids
    
    if (not _neutralize_charges) and (id_match_type == "same_connectivity"):
        # May not always cause the search to work, but should work for most molecules if the original one failed
        # The goal is to match the charge state to that of PubChem
        smiles = smiles.replace('[N+]=[N-]', 'z+').replace('[N-]=[N+]', 'z-') # replace azide with z+ and z-
        smiles = smiles.replace('O-', 'O').replace('[O]','O').replace('NH+','N').replace('NH2+', 'NH').replace('NH3+','NH2').replace('nH+', 'n').replace('N-','N').replace('[N]','N').replace('[n]','n').replace('S-','S').replace('[S]','S')
        smiles = smiles.replace('z+', '[N+]=[N-]').replace('z-', '[N-]=[N+]') # replace z+ and z- with azide
        return get_cids(smiles, _neutralize_charges = True)

    return []

def get_patents_from_cids(cids):
    """
    Get patent IDs (if available) for a list of PubChem CIDs.

    Parameters:
    -----------
    cids : list
        List of PubChem CIDs.
    
    Returns:
    --------
    patents : list
        List of patent IDs.
    """

    patents = []
    for cid in cids:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/PatentID/txt'
        response = pub_api_call(url = url)
        if response.ok and response.text != '':
            patents += [patent_id for patent_id in response.text.split('\n') if patent_id != '']    
    return patents

def patent_coverage(patent_ids, gpt_max_patents_per_mol=10):
    """
    Calculates the patent coverage of a list of patent IDs (ratio of patents checked out of total)

    Parameters:
    -----------
    patent_ids : list
        List of patent IDs.
    
    gpt_max_patents_per_mol : int
        Maximum number of patents to check per molecule. If -1, checks all patents.
    
    Returns:
    --------
    patent_coverage : float
        Ratio of patents checked out of total.
    """

    if len(patent_ids) == 0:
        return 0
    if gpt_max_patents_per_mol != -1:
        if len(patent_ids) > gpt_max_patents_per_mol:
            return  gpt_max_patents_per_mol / len(patent_ids)
        return 1
    return 1


# def get_patent_info(patent_ids, gpt_max_patents_per_mol:int=10):
#     """
#     Gets patent abstracts and descriptions for each patent in a list of patent IDs.

#     Scrapes the patent abstract and description from Google Patents.
    
#     Parameters:
#     -----------
#     patent_ids : list
#         List of patent IDs.

#     gpt_max_patents_per_mol : int
#         Maximum number of patents to check per molecule. If -1, checks all patents.

#     Returns:
#     --------
#     patent_info : dict
#         Dictionary of patent IDs to patent abstracts and descriptions. Format is {patent_id: {"abstract": abstract, "description": description}}.
#     """
    
#     patent_info = dict()
#     patents_scanned = 0
#     if len(patent_ids) > 0:
#         for patent_id in patent_ids:
#             try:
#                 url = f'https://patents.google.com/patent/{patent_id}/en'
#                 response = requests.get(url)
#                 if response.ok:
#                     # Scrape patent abstract and description
#                     html_content = response.text
#                     soup = BeautifulSoup(html_content, 'html.parser')
#                     abstract = soup.find('div', class_='abstract')
#                     description = soup.find('section', {'itemprop': 'description'})

#                     if abstract is None:
#                         abstract = "None"
#                     else:
#                         abstract = abstract.text.replace('\n', ' ')

#                     if description is None:
#                         description = "None"
#                     else:
#                         description = description.text.replace('\n', ' ')
                    
#                     patent_info.update({patent_id: {"abstract": abstract, "description": description}})
#                     patents_scanned += 1
#                     if patents_scanned >= gpt_max_patents_per_mol: # break if limit reached
#                         break
#             except Exception as e:
#                 print(patent_id, e)
#     return patent_info

# main for this file is deprecated. See pipeline.py
def main(args):
    start = time.time()

    # Unpack args
    input_ = args.input
    input_type = args.input_type
    head_n = args.head_n
    id_match_type = args.id_match_type
    gpt_key = args.api_key
    gpt_model = args.gpt_model
    gpt_system_prompt = args.gpt_system_prompt
    gpt_user_prompt = args.gpt_user_prompt
    desc_len = args.desc_len
    gpt_max_patents_per_mol = args.gpt_max_patents_per_mol
    gpt_temperature = args.gpt_temp



    # # check if args output_path is empty:
    # if args.output is None:
    #     output_path = f"{input_.split('/')[-1].split('.')[0]}_summarization.csv"
    # print(f"Output Path: {output_path}")
       

    output_dir = "surechembl_summarizations"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f"{output_dir}/top_1k.csv"
    df = load_data(input_, input_type, head_n)

    n_cpus = 6
    with mp.Pool(n_cpus) as p:
        print("INFO: Getting cids...")
        cids = p.map(get_cids, df['smiles'].tolist())
        print("INFO: Getting patent ids...")
        patent_ids = p.map(get_patents_from_cids, cids)
        patent_ids = p.map(set, patent_ids)
        coverage = p.map(patent_coverage, patent_ids)
        print("INFO: Getting patent info...")
        patent_info = p.starmap(get_patent_info, zip(patent_ids, repeat(gpt_max_patents_per_mol)))
        print(f"INFO: Summarizing patents with {gpt_model}...")
        summarizations_sources = p.starmap(summarization_wrapper, zip(patent_info, repeat(gpt_key), repeat(gpt_model), repeat(gpt_system_prompt), repeat(gpt_user_prompt), repeat(desc_len), repeat(gpt_temperature), repeat(gpt_max_patents_per_mol)))
        summarizations = p.map(summarizations_to_str, summarizations_sources)

    df['cids'] = [c for c in cids]
    df['patent_ids'] = [p for p in patent_ids]
    df['coverage'] = coverage
    df['patent_info'] = [p for p in patent_info]
    df['summarizations_sources'] = [s for s in summarizations_sources]
    df['summarizations'] = [s for s in summarizations]
    df[["smiles", "cids", "coverage", "summarizations_sources", "summarizations"]].to_csv(output_path, index=False)
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")
    print("Success!\n")

    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help="Input file path or single SMILES string, as specified by --input_type. If file, must be .csv with column labeled 'SMILES' or 'smiles'.")
    parser.add_argument('-o', '--output', type=str, required=False, help="Output file path")
    parser.add_argument('-k', '--api_key', type=str, required=True, help="OpenAI API key")
    parser.add_argument('-t', '--input_type', type=str, default="csv", choices = ("csv", "smiles", "SMILES"), help="Input type")
    parser.add_argument('-n', '--head_n', type=int, default=20, help="Number of top entries to run on if input is a file. Default 20 to avoid racking up API costs. Use '0' to select the entire file.")
    parser.add_argument('--desc_len', type=int, default=1000, help="Character limit for description. Use '0' to select the entire description. BEWARE: This may rack up API costs and/or cause the model to error out due to token limits.")
    parser.add_argument('--gpt_model', type=str, default='gpt-3.5-turbo', help="OpenAI Model Choice")
    parser.add_argument('--gpt_system_prompt', type=str, default = "You summarize chemical patents.", help="System prompt for GPT")
    parser.add_argument('--gpt_user_prompt', type=str, default = "Output a maximum of 1-2 words that summarizes the function of the molecule described by the following abstract and description:", help="User prompt for GPT")
    parser.add_argument('--gpt_max_patents_per_mol', type=int, default=10, help="Maximum number of patents to summarize per molecule. Default 10 to avoid racking up API costs. Use '-1' to set no limit.")
    parser.add_argument('--gpt_temp', type=float, default=0, help="Temperature for GPT (0-2). Higher values will increase randomness of output.")
    parser.add_argument('--id_match_type', type=str, default="same_connectivity", choices= ("same_connectivity", "exact_match"), help="Type of ID search to do when finding PubChem CIDs from SMILES. 'same_connectivity' will return all stereoisomers, while 'exact_match' will only match exact stereochemistry.")

    args = parser.parse_args()

    main(args)