import time
import openai

def chatgpt_functional_similarity(api_key, model = "gpt-3.5-turbo", system_prompt = "", user_prompt = "", descriptors_a = "", descriptors_b = "", gpt_temperature=0):
    """
    Determine functional similarity between two molecules using OpenAI's GPT API.

    Parameters
    ----------
    api_key : str
        OpenAI API Key

    model : str, optional
        OpenAI GPT Model. The default is "gpt-3.5-turbo".

    system_prompt : str
        System prompt for OpenAI model.

    user_prompt : str
        User prompt for OpenAI model.

    descriptors_a : str
        Functional descriptors for molecule A.

    descriptors_b : str
        Functional descriptors for molecule B.

    gpt_temperature : float, optional
        Temperature for OpenAI model. The default is 0.

    Returns
    -------
    str
        Functional similarity between molecules A and B.
    
    """
    
    
    if (api_key == ""):
        return "NA --- Missing API Key"
    if (descriptors_a in ["", "set()", r"{}"]):
        return "NA --- No descriptors provided for molecule A"
    if (descriptors_b in ["", "set()", r"{}"]):
        return "NA --- No descriptors provided for molecule B"
    if (descriptors_b == "API REQUEST ERROR"):
        return "NA --- API REQUEST ERROR ON PREVIOUS SUMMARIZATION STEP"
    api_success = False
    tries = 0
    #replace curly brackets in descriptors with nothing if they exist
    descriptors_a = descriptors_a.replace("{", "").replace("}", "")
    descriptors_b = descriptors_b.replace("{", "").replace("}", "")

    if system_prompt == "":
        system_prompt = "You are a chemistry expert."

    if user_prompt == "":
        prompt = f"The following are functional descriptors for Molecule A: [{descriptors_a}]\n And the following are functional descriptors for Molecule B: [{descriptors_b}]\nIs there at least one common function between these two molecules? You are required to reply in the format: '(yes or no) --- (20 words maximum explanation)'."
    else:
        prompt = f"{user_prompt}\nMolecule A: [{descriptors_a}]\nMolecule B: [{descriptors_b}]"

    while api_success == False:
        try:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=model,
                temperature=gpt_temperature,
                messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            time.sleep(5)
            tries += 1
            if tries > 10:
                print("ERROR: FAILED API REQUEST 10 TIMES. GIVING UP FOR THIS MOLECULE.")
                return "API REQUEST ERROR"
            else:
                pass



# main for this file is deprecated. See pipeline.py
# def main(args):
#     # pandarallel.initialize(progress_bar=False)

#     # Unpack args
#     input_file = args.input
#     query_file = args.query
#     output_file = args.output
#     api_key = args.api_key
#     gpt_model = args.gpt_model
#     gpt_system_prompt = args.gpt_system_prompt
#     gpt_user_prompt = args.gpt_user_prompt
#     gpt_temperature = args.gpt_temp

#     print('Loading data...')
#     query_df = pd.read_csv(query_file)
#     df = pd.read_csv(input_file)
#     print('Determining functional similarity...')
#     df["functional_similarity"] = df.apply(lambda x: chatgpt_functional_similarity(api_key, gpt_model, gpt_system_prompt, gpt_user_prompt, query_df["summarizations"][0], x["summarizations"], gpt_temperature), axis=1)
#     print('Saving data...')
#     df.to_csv(output_file, index=False)
#     print('Success!')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--input', type=str, required=True, help="Input file path to molecules with summarizations. Must include df column with 'summarizations'")
#     parser.add_argument('-q', '--query', type=str, required=True, help="Input file path to query molecule with summarizations, upon which to determine functional similarity. Must include df column with 'summarizations'")
#     parser.add_argument('-o', '--output', type=str, default= "./output.csv", required=False, help="Output file path")
#     parser.add_argument('-k', '--api_key', type=str, required=True, help="OpenAI API key")
#     parser.add_argument('--gpt_model', type=str, default='gpt-3.5-turbo', help="OpenAI Model Choice")
#     parser.add_argument('--gpt_system_prompt', type=str, default = "You are a chemistry expert.", help="System prompt for GPT")
#     parser.add_argument('--gpt_user_prompt', type=str, default = "", help="User prompt for GPT")
#     parser.add_argument('--gpt_temp', type=float, default=0, help="Temperature for GPT (0-2). Higher values will increase randomness of output.")

#     args = parser.parse_args()

#     main(args)