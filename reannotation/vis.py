import json
import torch
import cv2
import numpy

with open('./config.json', 'r') as f:
    config = json.load(f)

threshold = config['label']['threshold']
actions = config['label']['actions']
default_action = config['label']['default_action']
clip_duration = config['localization']['clip_duration']

time_step = config['localization']['time_step']
num = clip_duration // time_step

class Clip:
    def __init__(self, tracklet, action, logit):
        self.tracklet = tracklet
        self.action = action
        self.logit = logit


# 学生听课和老师讲课是默认动作，无需特别标注
def get_initial_action(logits, character):
    logit = torch.max(logits)
    if logit < threshold:
        return default_action[character], logit
    return actions[character][torch.argmax(logits)], logit

def get_final_action(clips):
    clip = max(clips, key=lambda x: x.logit)
    return clip.action


def aggregation(clips_file):
    clips = []
    with open(clips_file, 'r') as f:
        # Spatial aggregation
        clip = []
        for line in f:
            line = line.strip()
            if line:
                numbers = [num for num in line.split(",")]
                clip.append(numbers)
            else:
                clips.append(clip)
                clip = []
        if clip:
            clips.append(clip)
    # print(len(clips))
    # print(clips[56])
