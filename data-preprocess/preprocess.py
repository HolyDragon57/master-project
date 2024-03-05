import os
import json
import cv2
import math
from datetime import datetime
import numpy as np

# Macros: file paths, labels, etc.

video_path = "annotated_data/嵌套序列 11.mp4"
annotation_path = "annotated_data/11-1.json"
track_path = "ByteTrack/output/yolox_m_mix_det/track_result/2024_01_19_12_55_21.txt"
# index[0]占位符
index = ['', '老师', '表情', '学生1', '学生2', '学生3', '学生4', '学生5', '学生6', '学生7', '物品']
teacher_actions = ['讲课', '提问', '板书', '巡查', '回答', '操作电脑', '坐下', '喝水']
student_actions = ['看书', '记笔记', '坐着回答', '与同学交流', '拿物品', '举手', '起立回答', '伸懒腰', 
            '趴桌上', '走上讲台讲解', '吃东西', '喝水', '鼓掌', '传递物品', '打闹', '提问', '看电子产品']

# ByteTrack processing

# Extract target person and action from annotation json file

class Segment:
    def __init__(self, duration, label):
        self.startTime = duration[0]
        self.endTime = duration[1]
        self.character = next(iter(label))
        self.description = label[self.character]

def get_segments(annotation_path, track_path):
    with open(annotation_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    track_data = []
    with open(track_path, 'r', encoding='utf-8') as track_file:
        for line in track_file:
            numbers = [float(num) for num in line.strip().split(",")]
            track_data.append(numbers)

    metadata = data['metadata']
    
    segments = []
    for _, info in enumerate(metadata):
        if len(metadata[info]['xy']) == 0:
            segments.append(Segment(metadata[info]['z'], metadata[info]['av']))

    return metadata, track_data, segments

def confirm_id(segments, metadata, track_data):
    id = set()
    action_index = set([1, 3, 4, 5, 6, 7, 8, 9])
    for segment in segments:
        j = int(segment.character)
        if j not in id and j in action_index:
            min_diff = float('inf')  
            nearest = None
            for index, info in enumerate(metadata):
                if len(metadata[info]['xy']) != 0 and metadata[info]['z'][0] != 0:
                    diff = abs(segment.startTime - metadata[info]['z'][0])
                    if diff < min_diff:
                        min_diff = diff
                        nearest = info
            frame = int(metadata[info]['z'][0] * 30)
            targetID = -1
            for tracklet in track_data:
                IoU = 0
                target = metadata[nearest]['xy'][1:]
                if int(tracklet[0]) == frame:
                    tmp = calculate_iou(target, tracklet[2:6])
                    if tmp > IoU:
                        IoU = tmp
                        targetID = tracklet[1]
                if tracklet[0] > frame:
                    break
            if targetID != -1:
                id.add(targetID)
    return id

def extract_target(id, track_data):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video_name = 'extraction_result_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        for tracklet in track_data:
            if tracklet[1] in id:
                color = assign_color(int(tracklet[1]))
                cv2.rectangle(frame, 
                    (int(tracklet[2]), int(tracklet[3])), 
                    (int(tracklet[2] + tracklet[4]), int(tracklet[3] + tracklet[5])),
                    color,
                    2)

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord("Q") or key == 27:
            break

        out.write(frame)
    
    cap.release()        

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
    x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    iou = inter_area / union_area

    return iou

# Produce isolated training data

def assign_color(cls_id): 
    _COLORS = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.314, 0.717, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32).reshape(-1, 3)

    return (_COLORS[cls_id % 80] * 255).astype(np.uint8).tolist()

metadata, track_data, segments = get_segments(annotation_path, track_path)
id = confirm_id(segments, metadata, track_data)
print(id)
# for segment in segments:
#     attrs = vars(segment)
#     print(', '.join("%s: %s" % item for item in attrs.items()))
extract_target(id, track_data)