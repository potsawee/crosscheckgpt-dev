import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

def main(
    temperature,
    num_samples=5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    print("model:", model_name)
    print("temperature:", temperature)
    print("num_samples:", num_samples)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"]
    wikibio = load_dataset("wiki_bio", split="test")

    outputs = []
    for i in tqdm(range(len(dataset))):
        entity = wikibio[dataset[i]['wiki_bio_test_idx']]['input_text']['context'].replace("-lrb-", "(").replace("-rrb-", ")").strip()
        
        prompt = f"Write a Wikipedia about {entity} for around 200 words\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        samples = []        
        for _ in range(num_samples):
            generate_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=temperature,
            )
            output_text = tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            generate_text = output_text.replace(prompt, "")
            samples.append(generate_text)
        
        outputs.append(samples)
        
    with open(f'outputs/Mistral-7B-Instruct-v0.2-temp{temperature}.json', 'w') as fout:
        json.dump(outputs , fout)

def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--temperature', type=float, required=True)
    return parser

with torch.no_grad():
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    with torch.no_grad():
        main(**kwargs)

