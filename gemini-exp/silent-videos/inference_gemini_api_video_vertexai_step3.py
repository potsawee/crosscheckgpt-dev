from pathlib import Path
from glob import glob
from tqdm import tqdm
import os
import json
import random
import hashlib
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import time


project_id = "311305667538"

# Set up the model
generation_config = {
  "temperature": 0.0,
  "max_output_tokens": 4096,
}


def exp():
  model = GenerativeModel(model_name="gemini-1.5-pro-preview-0409",
                                generation_config=generation_config)

  prompt = "Describe the audio in this video in one paragraph."
  paths = glob("normal_videos/*.mp4")
  random.shuffle(paths)


  for file_path in tqdm(paths):
    file_name = file_path.replace("normal_videos/", "").replace(".mp4", "")
    video_file_uri = (
        f"gs://crosscheck-videos/videos/{file_name}.mp4"
    )
    video_file = Part.from_uri(video_file_uri, mime_type="video/mp4")


    output_path = f"outputs_step3/{file_name}.json"
    if os.path.isfile(output_path):
      print("file exist:", output_path)
      continue

    response = model.generate_content([video_file, prompt])
    try:
      print(response.text)
    except:
      import ipdb; ipdb.set_trace()
      
    with open(output_path, "w") as f:
      f.write(response.text)
    print("write:", output_path)

    time.sleep(15) # Requests limited to 2 per second


for i in range(1, 100):
  try:
    region = random.choice(["us-central1", "us-west1", "us-west4", "us-east4"])
    print(region)
    vertexai.init(project=project_id, location=region)
    exp()
  except:
    print(f"ERROR #{i}..waiting 15 sec")
    time.sleep(30)
