import difflib
import json
import ipdb

path0 = "annotations_with_repetition_215.json"
path1 = "annotations_with_repetition_215_gpt_clean.json"
with open(path0) as f:
    x = json.load(f)
with open(path1) as f:
    y = json.load(f)

for i in range(215):
    assert x[i]['video_id'] == y[i]['video_id']
    print("i=", i)
    if x[i]['audio_description'] == y[i]['audio_description']:
        print("[/] audio matched!!")
    else:
        print("[---------- audio diff ----------]")
        difference = difflib.Differ()
        #Calculates the difference
        diff = difference.compare([x[i]['audio_description']],[y[i]['audio_description']])
        print ('\n'.join(diff))

    if x[i]['visual_description'] == y[i]['visual_description']:
        print("[/] visual matched!!")
    else:
        print("[---------- visual diff ----------]")
        difference = difflib.Differ()
        #Calculates the difference
        diff = difference.compare([x[i]['visual_description']],[y[i]['visual_description']])
        print ('\n'.join(diff))
    ipdb.set_trace()