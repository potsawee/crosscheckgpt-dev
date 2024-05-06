from pathlib import Path
from glob import glob
from tqdm import tqdm
import os
import random
import hashlib
import google.generativeai as genai
import time

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 4096,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]

def exp():
  model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                generation_config=generation_config,
                                safety_settings=safety_settings)

  prompt = "Describe the audio in one paragraph."


  paths = glob("audios/*.wav")
  # random.shuffle(paths)

  for file_path in tqdm(paths):
    file_name = file_path.replace("audios/", "").replace(".wav", "")
    output_path = f"outputs/{file_name}.txt"
    if os.path.isfile(output_path):
      print("file exist:", output_path)
      continue

    wav_file = genai.upload_file(path=file_path)

    response = model.generate_content([wav_file, prompt])
    print(response.text)
    print(model.count_tokens([wav_file]))

    with open(output_path, "w") as f:
      f.write(response.text)
    print("write:", output_path)
    time.sleep(10) # Requests limited to 2 per second


for i in range(1, 1000):
  try:
    exp()
  except:
    print(f"ERROR #{i}: google.api_core.exceptions.ResourceExhausted")
    time.sleep(60)

exp()