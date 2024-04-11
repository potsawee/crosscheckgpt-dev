import os
import openai
from openai import OpenAI

import argparse
import json
import torch
from tqdm import tqdm

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    # organization=os.getenv("OPENAI_ORGANIZATION"),
)

def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", bool, lambda v: v.lower() == "true")
    parser.add_argument('--model', type=str, default='gpt-4-turbo-2024-04-09')
    parser.add_argument('--onebest_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    model = kwargs['model']
    onebest_path = kwargs['onebest_path']
    output_dir = kwargs['output_dir']

    print("model:", model)
    print("onebest_path:", onebest_path)
    print("output_dir:", output_dir)

    # LLM to be evaluated
    with open(onebest_path, "r") as f:
        onebest = json.load(f)
    # onebest[i] -> dict(image_name: ..., response: ...)

    system_prompt = "You are a helpful assistant that help break a passage into independent facts. You must return the facts in a list\n\n"
    system_prompt += "This is an example of how you break a passage.\n\n"
    system_prompt += "Passage: He made his acting debut in the film The Moon is the Sun’s Dream (1992), and continued to appear in small and supporting roles throughout the 1990s.\n\n"
    system_prompt += "Facts: [\n"
    system_prompt += "'He made his acting debut in the film.',\n"
    system_prompt += "'He made his acting debut in The Moon is the Sun’s Dream.',\n"
    system_prompt += "'The Moon is the Sun’s Dream is a film.',\n"
    system_prompt += "'The Moon is the Sun’s Dream was released in 1992.',\n"
    system_prompt += "'After his acting debut, he appeared in small and supporting roles.',\n"
    system_prompt += "'After his acting debut, he appeared in small and supporting roles throughout the 1990s.',\n"
    system_prompt += "]"
    

    for i in range(len(onebest)):
        output_path = f"{output_dir}/{i}.json"
        if os.path.isfile(output_path):
            print("{}: already exists".format(output_path))
            continue        
        passage = onebest[i]['response']
        prompt = f"Passage: {passage}\n\nFacts:"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        gen_text = response.choices[0].message.content
        item = {
            'i': i,
            'image_name': onebest[i]['image_name'],
            'onebest_path': onebest_path,
            'facts': gen_text,
        }
        with open(output_path, "w") as f:
            json.dump(item, f)
        print("write:", output_path)
        