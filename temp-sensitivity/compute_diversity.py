import json
import argparse
import torch
import spacy
from tqdm import tqdm
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt

def main(
    model, # Mistral-7B-Instruct-v0.2
    temperature
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selfcheck_prompt = SelfCheckLLMPrompt("mistralai/Mistral-7B-Instruct-v0.2", device)
    nlp = spacy.load("en_core_web_sm")
    # sentences = [sent for sent in nlp(passage).sents] # List[spacy.tokens.span.Span]
    # sentences = [sent.text.strip() for sent in sentences if len(sent) > 3]


    print("model:", model)
    print("temperature:", temperature)

    # one-best output
    with open(f'outputs/{model}-temp0.0.json', 'r') as f:
        onebest = json.load(f)   
    # sampled outputs at temperature
    with open(f'outputs/{model}-temp{temperature}.json', 'r') as f:
        samples = json.load(f) 

    assert len(onebest) == len(samples)

    scores = []
    for i in tqdm(range(len(onebest))):
        text = onebest[i][0] # str
        sentences = [sent for sent in nlp(text).sents] # List[spacy.tokens.span.Span]
        sentences = [sent.text.strip() for sent in sentences if len(sent) > 3]
        evidences = samples[i] # List[str]

        sent_scores_prompt = selfcheck_prompt.predict(
            sentences = sentences, # list of sentences
            sampled_passages = evidences, # list of sampled passages
            verbose = False, # whether to show a progress bar
        )
        scores.append({
            'i': i,
            'scores': sent_scores_prompt.tolist(),
            'method': 'llm-prompt-mistral'
        })
        
    with open(f'outputs/scoring/{model}-temp{temperature}.json', 'w') as fout:
        json.dump(scores , fout)

def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    return parser

with torch.no_grad():
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    with torch.no_grad():
        main(**kwargs)

