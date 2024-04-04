import cv2
import numpy as np
import torch
import json

with open('./config.json', 'r') as f:
    config = json.load(f)

teacher_actions = config['label']['actions']['teacher']
student_actions = config['label']['actions']['student']
student_actions_en = config['label']['actions']['student_en']

def tracklet_to_video(tracklet):
    

def video_to_tensor(video_path, img_size=320, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter("test.mp4", fourcc, 1, (img_size, img_size))
    
    video_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (img_size, img_size))
            video_frames.append(frame)
            out.write(frame)
    cap.release()
    
    video_tensor = torch.tensor(np.array(video_frames), dtype=torch.float32).permute(0, 3, 1, 2)
    video_tensor = video_tensor.unsqueeze(0)
    print(video_tensor)
    
    return video_tensor

def label_mapping():
    return _ 

def action_recognition(video_tensor=None):
    random_vector = torch.rand((17,))
    logits = torch.softmax(random_vector, dim=0)
    print(student_actions[torch.argmax(logits)])
    return logits

# video_to_tensor("../temporal-action-localization/output_clips/demo_clip_5.mp4")
action_recognition()