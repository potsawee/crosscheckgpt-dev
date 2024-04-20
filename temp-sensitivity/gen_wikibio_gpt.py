import os
from openai import OpenAI
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

def main(
    model,
    output_path,
    temperature=0.0,
    num_samples=1,
):

    print("model:", model) # gpt-3.5-turbo-0125, gpt-4-turbo-2024-04-09
    print("output_path:", output_path)
    print("temperature:", temperature)
    print("num_samples:", num_samples)

    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"]
    wikibio = load_dataset("wiki_bio", split="test")
    outputs = []
    for i in tqdm(range(len(dataset))):
        entity = wikibio[dataset[i]['wiki_bio_test_idx']]['input_text']['context'].replace("-lrb-", "(").replace("-rrb-", ")").strip()
        prompt = f"Generate a passage about {entity}.\n\n"
        samples = []        
        for _ in range(num_samples):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=4096,
            )
            gen_text = response.choices[0].message.content
            samples.append(gen_text)
        outputs.append(samples)
    with open(output_path, 'w') as fout:
        json.dump(outputs , fout)


def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--model', type=str, default='gpt-4-turbo-2024-04-09')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num_samples', type=int, default=1)
    return parser

parser = argparse.ArgumentParser()
parser = add_arguments(parser)
kwargs = vars(parser.parse_args())
main(**kwargs)

