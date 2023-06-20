python patent_summarization.py\
       -i "../../data/surechembl/surechembl_smiles.csv"\
       -t "csv"\
       -k $OPENAI_KEY\
       -n 100\
       --desc_len 3500\
       --gpt_model "gpt-3.5-turbo"\
       --gpt_system_prompt "You are an organic chemist summarizing chemical patents"\
       --gpt_user_prompt "Output a brief list of 1-3 word phrases that best describe the chemical and pharmacological function(s) of the molecule described by the given patent abstract and description. Only use mechanistic descriptors (eg. x inhibitor, y ligands, z agonist, phosphorescent, etc), and don't use structural descriptors (peptide derivatives, halogenated, etc). Be specific, concise, and follow the syntax 'descriptor_1 / descriptor_2 / etc', writing 'NA' if nothing is provided. The following is the patent: "\
       --gpt_max_patents_per_mol 10\
       --id_match_type "exact_match"\
       --gpt_temp 0\

       # --gpt_user_prompt "Output a list of 1-3 word phrases that best describe the chemical and pharmacological function(s) of a molecule described by a patent abstract and description. Be specific, and follow the syntax 'descriptor_1 / descriptor_2 / etc'. The following is the patent: "\
       # -i "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"\
       # -t "smiles"\
       # -o "penicillin_g.csv"\
       # --gpt_user_prompt "Output a max of 1-2 words that summarizes the chemical function of a molecule described by a patent abstract and description. Try to be specific rather than general. The following is the patent: "\