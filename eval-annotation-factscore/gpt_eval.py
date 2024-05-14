import os
import openai
from openai import OpenAI
import random
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

system_prompt = "You are a helpful assistant that helps evaluate a sentence with respect to a context. You must answer whether the sentence is supported by the context (Support means most of the information is supported), contradict the context (Contradict means most of the information is not supported), or Neutral meaning that the information is neither supported nor contradicted (i.e., the information is mostly irrelevant).\n\n"

system_prompt += "Your answer must be one of the following options (not no further explanation): Support, Contradict, Neutral.\n\n"
system_prompt += "These are some examples.\n\n"
system_prompt += "Sentence: The Indian guy's voice talks about Vilayat Khan's style of sitar.\n"
system_prompt += "Context: The man is talking about the skills of the sitar. There is no music playing.\n"
system_prompt += "Answer: Support\n\n"

system_prompt += "Sentence: The melody is simple.\n"
system_prompt += "Context: The audio plays the sound of a xylophone.\n"
system_prompt += "Answer: Neutral\n\n"

system_prompt += "Sentence: The young woman is seated on a green sofa.\n"
system_prompt += "Context: The video depicts a young woman as she performs a song on a ukulele while seated comfortably on a gray sofa in a bright and elegantly decorated living room. She begins her performance with a gentle smile, her fingers positioned on the ukulele's strings and fretboard. The piece is from Bruno Mars's \"Count on Me\".\n"
system_prompt += "Answer: Contradict\n\n"



def call_api(sentence, context):
    prompt = system_prompt 
    prompt += f"Sentence: {sentence}\n"
    prompt += f"Context: {context}\n"
    prompt += "Answer:"
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
    return gen_text
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    model = kwargs['model']
    output_dir = kwargs['output_dir']

    print("model:", model)
    print("output_dir:", output_dir)

    ids = [i for i in range(39)]
    random.shuffle(ids)
    for i in ids:
        print("working on i=", i)
        output_path = f"{output_dir}/{i}.json"
        if os.path.isfile(output_path):
            print("{}: already exists".format(output_path))
            continue 
        with open(f"/data/workspace/exp-punpun/crosscheckgpt-dev/atomic-facts/outputs/annotations_double/{i}.json", "r") as f:
            annotation = json.load(f)
        annotation0 = annotation[0]
        annotation1 = annotation[1]

        # ann0 checked by ann1
        ann0_audio_facts = eval(annotation0['audio_facts'])
        ann0_audio_factscores = []
        for ann0_audio_fact in ann0_audio_facts:
            ann0_audio_factscores += [call_api(ann0_audio_fact, annotation1['audio_description'])]

        ann0_visual_facts = eval(annotation0['visual_facts'])
        ann0_visual_factscores = []
        for ann0_visual_fact in ann0_visual_facts:
            ann0_visual_factscores += [call_api(ann0_visual_fact, annotation1['visual_description'])]

        # ann1 checked by ann0
        ann1_audio_facts = eval(annotation1['audio_facts'])
        ann1_audio_factscores = []
        for ann1_audio_fact in ann1_audio_facts:
            ann1_audio_factscores += [call_api(ann1_audio_fact, annotation0['audio_description'])]

        ann1_visual_facts = eval(annotation1['visual_facts'])
        ann1_visual_factscores = []
        for ann1_visual_fact in ann1_visual_facts:
            ann1_visual_factscores += [call_api(ann1_visual_fact, annotation0['visual_description'])]

        item = {
            'annotation': annotation,
            'eval': {
                'ann0_audio_factscores': ann0_audio_factscores,
                'ann0_visual_factscores': ann0_visual_factscores,
                'ann1_audio_factscores': ann1_audio_factscores,
                'ann1_visual_factscores': ann1_visual_factscores,
            }
        }
        with open(output_path, "w") as f:
            json.dump(item, f)
        print("write:", output_path)

        