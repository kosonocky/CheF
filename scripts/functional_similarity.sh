python functional_similarity.py\
       -i "lsd_rdkit_atom_0_summarization_2.csv"\
       -q "lsd_manual.csv"\
       -o "lsd_rdkit_atom_0_functional_similarity_2.csv"\
       -k $OPENAI_KEY\
       --gpt_model "gpt-3.5-turbo"\
       --gpt_system_prompt "You are a chemistry expert."\
       --gpt_user_prompt "Given functional descriptors of two molecules, determine if the two molecules have similar function. If similar elements are found in both lists, these molecules have similar function. You must respond in the format '{yes or no} --- {20 word maximum explanation}'"\