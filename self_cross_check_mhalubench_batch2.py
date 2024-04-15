import argparse
import os
import json
import spacy
import numpy as np
import torch
from tqdm import tqdm
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt

def main(
    llm_evaluated,
    temperature,
):
    print("llm_evaluated:", llm_evaluated)
    print("temperature:", temperature)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selfcheck_prompt = SelfCheckLLMPrompt("mistralai/Mistral-7B-Instruct-v0.2", device)
    nlp = spacy.load("en_core_web_sm")
    # sentences = [sent for sent in nlp(passage).sents] # List[spacy.tokens.span.Span]
    # sentences = [sent.text.strip() for sent in sentences if len(sent) > 3]

    # LLM to be evaluated
    with open(f"mhalubench/OneBest/{llm_evaluated}_mhalubench_onebest.json", "r") as f:
        onebest = json.load(f)
    onebest_dict = {} # image_name -> response
    for x in onebest:
        onebest_dict[x["image_name"]] = x["response"]

    # LLM for evidence generation (selfcheck & crosscheck-explicit)
    evidence_llms = ["avsalmonn", "llamavid", "valley", "chat_univi", "llava1.5_7b",
                     "videollama", "instructblip", "mplug_owl2", "videollava"]

    for evidence_llm in evidence_llms:
        print(f"LLM={llm_evaluated}, Evidence={evidence_llm}")
        output_path = f'outputs/mhalubench_batch2_temp{temperature}/{llm_evaluated}_{evidence_llm}.json'
        if os.path.isfile(output_path):
            print("{}: already exists".format(output_path))
            continue    

        outputs = []        
        with open(f"mhalubench/Passages_Temperature/{evidence_llm}_temp{temperature}.json", "r") as f:
            passages = json.load(f)
        passages_dict = {} # image_name -> response
        for x in passages:
            passages_dict[x["image_name"]] = x["response"]

        for image_name in tqdm(onebest_dict.keys()): # 499
            text = onebest_dict[image_name] # str
            sentences = [sent for sent in nlp(text).sents] # List[spacy.tokens.span.Span]
            sentences = [sent.text.strip() for sent in sentences if len(sent) > 3]
            evidences = passages_dict[image_name] # List[str]

            if len(sentences) == 0:
                sentences = ["."]

            sent_scores_prompt = selfcheck_prompt.predict(
                sentences = sentences, # list of sentences
                sampled_passages = evidences, # list of sampled passages
                verbose = False, # whether to show a progress bar
            )
            outputs.append({
                'image_name': image_name,
                'llm_gen_text': llm_evaluated,
                'llm_gen_evidence': evidence_llm,
                'scores': sent_scores_prompt.tolist(),
                'method': 'llm-prompt-mistral'
            })

        with open(output_path, 'w') as fout:
            json.dump(outputs , fout)
        print(f"completed: {llm_evaluated}")
            
def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--llm_evaluated', type=str, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    with torch.no_grad():
        main(**kwargs)

