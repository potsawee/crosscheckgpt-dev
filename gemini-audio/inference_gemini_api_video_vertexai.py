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
vertexai.init(project=project_id, location="us-central1")

# Set up the model
generation_config = {
  "temperature": 1.0,
  "top_p": 0.95,
  "max_output_tokens": 4096,
}


def exp():
  model = GenerativeModel(model_name="gemini-1.5-pro-preview-0409",
                                generation_config=generation_config)

  prompt = "Describe the video (visual and audio) in one paragraph."
  paths = glob("videos/*.mp4")
  random.shuffle(paths)

  num_samples = 10

  for file_path in tqdm(paths):
    file_name = file_path.replace("videos/", "").replace(".mp4", "")
    video_file_uri = (
        f"gs://crosscheck-videos/videos/{file_name}.mp4"
    )
    video_file = Part.from_uri(video_file_uri, mime_type="video/mp4")


    for ns in range(num_samples):
      output_path = f"outputs_vertex_ai_video_samples10/{file_name}_sample{ns}.json"
      if os.path.isfile(output_path):
        print("file exist:", output_path)
        continue

      response = model.generate_content([video_file, prompt])
      print(response.text)

      with open(output_path, "w") as f:
        f.write(response.text)
      print("write:", output_path)

      time.sleep(10) # Requests limited to 2 per second


for i in range(1, 5000):
  try:
    exp()
  except:
    print(f"ERROR #{i}..waiting 15 sec")
    time.sleep(15)
