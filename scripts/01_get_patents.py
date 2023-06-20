import time
import requests
import argparse
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import multiprocessing as mp
import openai
from bs4 import BeautifulSoup
import sys
from pathlib import Path
from itertools import repeat


def load_data(input_:str, input_type:str = "csv") -> pd.DataFrame:
    """
    Loads data from a csv file or single SMILES string into a pandas dataframe.

    Parameters:
    -----------
    input_ : str
        Path to the input file or SMILES string.
    input_type : str
        Type of input. Either "csv" or "smiles".

    Returns:
    --------
    df : pandas dataframe
        Dataframe containing SMILES strings.

    """

    if input_type == "csv":
        df = pd.read_csv(input_)
        df = df.rename(columns={"SMILES": "smiles"})
    else:
        df = pd.DataFrame({"smiles": [input_]})
    return df


def pub_api_call(url, params = None, _n_restarts = 0):
    """
    Calls the PubChem API. If the API is overloaded, it will wait and try again.

    This function is specific to the PubChem API, as the timeout code is set to 503 instead of 429.

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



def get_cids(smiles, id_match_type = "exact_match", _neutralize_charges = False, ):
    """
    Gets PubChem CIDs given a SMILES string. Options to search exact match or same connectivity (stereoisomers).

    For same connectivity, if it cannot find a match, it will attempt to neutralize all charges according to PubChem's rules and try again.
    
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

def get_patent_ids_from_cids(cids):
    """
    Get patent IDs (if available) for a list of PubChem CIDs.

    Note: This could be optimized to directly add to set.

    Parameters:
    -----------
    cids : list
        List of PubChem CIDs.
    
    Returns:
    --------
    patents : set
        Set of patent IDs.
    """

    patents = []
    for cid in cids:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/PatentID/txt'
        response = pub_api_call(url = url)
        if response.ok and response.text != '':
            patents += [patent_id for patent_id in response.text.split('\n') if patent_id != '']    
    return set(patents)

def patent_coverage(patent_ids, max_patents_per_mol=10):
    """
    Calculates the patent coverage of a list of patent IDs (ratio of patents checked out of total)

    Parameters:
    -----------
    patent_ids : list
        List of patent IDs.
    
    max_patents_per_mol : int
        Maximum number of patents to check per molecule. If -1, checks all patents.
    
    Returns:
    --------
    patent_coverage : float
        Ratio of patents checked out of total.
    """

    if len(patent_ids) == 0:
        return 0
    if max_patents_per_mol != -1:
        if len(patent_ids) > max_patents_per_mol:
            return  max_patents_per_mol / len(patent_ids)
        return 1
    return 1

# main for this file is deprecated. See pipeline.py
def main(args):
    start = time.time()

    # Unpack args
    input_ = args.input
    input_type = args.input_type
    batch_size = args.batch_size
    max_patents_per_mol = args.max_patents_per_mol
    id_match_type = args.id_match_type


    # # check if args output_path is empty:
    # if args.output is None:
    #     output_path = f"{input_.split('/')[-1].split('.')[0]}_summarization.csv"
    # print(f"Output Path: {output_path}")
       

    output_dir = "../results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    main_df = load_data(input_, input_type)

    n_cpus = 6

    for count, i in enumerate(range(0, len(main_df), batch_size)):
        if i + batch_size > len(main_df):
            df = main_df[i:]
            print(f"\nINFO: Processing {i} to {len(main_df)}...")
        else:
            df = main_df[i:i+batch_size]
            print(f"\nINFO: Processing {i} to {i+batch_size}...")

        output_path = f"{output_dir}/schembl_pids_b{str(count).zfill(5)}_i{str(i).zfill(9)}.pkl"
        
        with mp.Pool(n_cpus) as p:
            print("INFO: Getting cids...")
            cids = p.starmap(get_cids, zip(df['smiles'].tolist(), repeat(id_match_type)))
            print("INFO: Getting patent ids...")
            patent_ids = p.map(get_patent_ids_from_cids, cids)
            print("INFO: Calculating patent coverage...")
            coverage = p.starmap(patent_coverage, zip(patent_ids, repeat(max_patents_per_mol)))

        df['cids'] = cids
        df['patent_ids'] = patent_ids
        df['coverage'] = coverage
    
        df = df[df["coverage"] == 1].reset_index(drop=True)
        df[["smiles", "cids", "patent_ids"]].to_pickle(output_path)
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")
    print("Success!\n")

    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help="Input file path or single SMILES string, as specified by --input_type. If file, must be .csv with column labeled 'SMILES' or 'smiles'.")
    parser.add_argument('-t', '--input_type', type=str, default="csv", choices = ("csv", "smiles", "SMILES"), help="Input type")
    parser.add_argument('-n', '--batch_size', type=int, default=20, help="Number of molecules per batch. Use '0' to select the entire file.")
    parser.add_argument('--max_patents_per_mol', type=int, default=10, help="Maximum number of patents to summarize per molecule. Default 10 to avoid racking up API costs. Use '-1' to set no limit.")
    parser.add_argument('--id_match_type', type=str, default="same_connectivity", choices= ("same_connectivity", "exact_match"), help="Type of ID search to do when finding PubChem CIDs from SMILES. 'same_connectivity' will return all stereoisomers, while 'exact_match' will only match exact stereochemistry.")

    args = parser.parse_args()

    main(args)