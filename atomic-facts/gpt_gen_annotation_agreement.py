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
    parser.add_argument('--output_dir', type=str, required=True)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    model = kwargs['model']
    output_dir = kwargs['output_dir']

    print("model:", model)
    print("output_dir:", output_dir)

    # LLM to be evaluated
    double_annotation_path = "/data/workspace/exp-punpun/crosscheckgpt-dev/annotation_audio_video/final_combine/annotations_double.json"
    with open(double_annotation_path, "r") as f:
        data = json.load(f)

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
    
    print("len(data):", len(data))

    for i in range(len(data)):
        output_path = f"{output_dir}/{i}.json"
        if os.path.isfile(output_path):
            print("{}: already exists".format(output_path))
            continue        
        fact_outputs = []
        for annotation in data[i]:
            audio_description = annotation['audio_description']
            visual_description = annotation['visual_description']

            prompt_audio = f"Passage: {audio_description}\n\nFacts:"
            prompt_visual = f"Passage: {visual_description}\n\nFacts:"
            response_audio = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_audio},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            response_visual = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_visual},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            facts_audio = response_audio.choices[0].message.content
            facts_visual = response_visual.choices[0].message.content
            fact_outputs.append({
                'video_id': annotation['video_id'],
                'audio_description': annotation['audio_description'],
                'visual_description': annotation['visual_description'],
                'audio_facts': facts_audio,
                'visual_facts': facts_visual,
            })
        with open(output_path, "w") as f:
            json.dump(fact_outputs, f, indent=4)
        print("write:", output_path)
        