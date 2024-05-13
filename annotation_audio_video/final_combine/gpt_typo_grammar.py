import os
import openai
from openai import OpenAI

import argparse
import json
import torch
from tqdm import tqdm
import difflib

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
    with open("./annotations_with_repetition_215.json", "r") as f:
        data = json.load(f)
    # onebest[i] -> dict(video_id: ..., audio_description: ..., visual_description: ...)

    system_prompt = "You are a helpful assistant that helps correct text. Your task is to correct any typo and grammatical errors. Please only make changes if it is a typo or grammatical error; otherwise, do not change the text. Below are some examples.\n\n"

    for i in range(len(data)):
        output_path = f"{output_dir}/{i}.json"
        if os.path.isfile(output_path):
            print("{}: already exists".format(output_path))
            continue        
        audio_description0 = data[i]['audio_description']
        visual_description0 = data[i]['visual_description']
        prompt_audio = f"Input: {audio_description0}\n\n Corrected Text (only make changes if there is a typo or grammatical error):"
        prompt_visual = f"Input: {visual_description0}\n\n Corrected Text (only make changes if there is a typo or grammatical error):"
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
        audio_description1 = response_audio.choices[0].message.content
        visual_description1 = response_visual.choices[0].message.content

        if audio_description0 == audio_description1:
            print("[/] audio matched!!")
        else:
            print("audio diff.......")
            difference = difflib.Differ()
            #Calculates the difference
            diff = difference.compare([audio_description0],[audio_description1])
            print ('\n'.join(diff))
            print("-------------------------------")

        if audio_description0 == audio_description1:
            print("[/] visual matched!!")
        else:
            print("visual diff.......")
            difference = difflib.Differ()
            #Calculates the difference
            diff = difference.compare([visual_description0],[visual_description1])
            print ('\n'.join(diff))
            print("-------------------------------")
        item = {
            'video_id': data[i]['video_id'],
            'audio_description': audio_description1,
            'visual_description': visual_description1,
        }
        with open(output_path, "w") as f:
            json.dump(item, f)
        print("write:", output_path)
        