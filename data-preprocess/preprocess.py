import os
import os.path as osp
import json
import cv2
import math
from datetime import datetime
import numpy as np
import csv

# Macros: file paths, labels, etc.

video_path = "annotated_data/嵌套序列 11.mp4"
file_name = osp.splitext(osp.basename(video_path))[0]
annotation_path = "annotated_data/11-1.json"
track_path = "annotated_data/嵌套序列 11_2024_03_06_15_27_54.txt"
output_dir = "extraction_result"
teacher_dataset_label_path = "extraction_result/teacher_clip-action.csv"
student_dataset_label_path = "extraction_result/student_clip-action.csv"
# index[0]占位符
index = ['', '老师', '表情', '学生1', '学生2', '学生3', '学生4', '学生5', '学生6', '学生7', '物品']
teacher_actions = ['讲课', '提问', '板书', '巡查', '回答', '操作电脑', '坐下', '喝水']
student_actions = ['看书', '记笔记', '坐着回答', '与同学交流', '拿物品', '举手', '起立回答', '伸懒腰', 
            '趴桌上', '走上讲台讲解', '吃东西', '喝水', '鼓掌', '传递物品', '打闹', '提问', '看电子产品']
action_index = set([1, 3, 4, 5, 6, 7, 8, 9])
# 由于人体和上半身的缘故，高度上可能可以截掉一半？
upper_body_ratio = 0.6

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
    bboxes = {}
    for i, info in enumerate(metadata):
        if len(metadata[info]['xy']) == 0:
            segments.append(Segment(metadata[info]['z'], metadata[info]['av']))
        else:
            if int(next(iter(metadata[info]['av']))) in action_index:
                bboxes[info] = metadata[info]

    return bboxes, track_data, segments

def confirm_id(segments, bboxes, track_data):
    segment_id = {}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_bboxes = {}
    for index, info in enumerate(bboxes):
        if bboxes[info]['z'][0] == 0:
            start_bboxes[info] = bboxes[info]

    for segment in segments:
        i = int(segment.character)
        assert i in action_index
        if segment.startTime != 0:
            min_diff = float('inf')  
            nearest = None
            for index, info in enumerate(bboxes):
                diff = abs(segment.startTime - bboxes[info]['z'][0])
                if diff < min_diff:
                    min_diff = diff
                    nearest = info
            frame = int(bboxes[nearest]['z'][0] * fps) - 1
            targetID = -1
            index = binary_search(track_data, frame)
            target = bboxes[nearest]['xy'][1:]
            IoU = 0
            while int(track_data[index][0]) == frame:
                tmp = calculate_iou(target, track_data[index][2:6])
                if tmp > IoU:
                    IoU = tmp
                    targetID = int(track_data[index][1])
                index += 1
            if targetID != -1 and IoU > 0.2:
                segment_id[segment] = targetID
        else:
            target = None
            for index, info in enumerate(start_bboxes):
                if int(segment.character) == int(next(iter(start_bboxes[info]['av']))):
                    target = start_bboxes[info]['xy'][1:]
                    break
            i = 0
            IoU = 0
            targetID = -1
            while int(track_data[i][0]) == 0:
                tmp = calculate_iou(target, track_data[i][2:6])
                if tmp > IoU:
                    IoU = tmp
                    targetID = int(track_data[i][1])
                i += 1
            if targetID != -1 and IoU > 0.2:
                segment_id[segment] = targetID

    return segment_id

def extract_target(segment_id, track_data):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    segment_num = 0

    for segment, id in segment_id.items():
        attrs = vars(segment)
        print(', '.join("%s: %s" % item for item in attrs.items()))

        # output_video_name = file_name + '_extraction_result_' + str(segment_num) + ".mp4"
        if int(segment.character) == 1:
            output_video_name = file_name + '_' + str(segment_num) + '-' + teacher_actions[int(segment.description)] + ".mp4"
        else:
            output_video_name = file_name + '_' + str(segment_num) + '-' +student_actions[int(segment.description)] + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(osp.join(output_dir, output_video_name), fourcc, fps, (width, height))
        print(f"Writing {output_video_name}")

        current_frame = math.floor(segment.startTime * fps) - 1 if segment.startTime != 0 else 0
        end_frame = math.ceil(segment.endTime * fps) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        index = binary_search(track_data, current_frame)
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            while index < len(track_data) and int(track_data[index][0]) == current_frame:
                if int(track_data[index][1]) == id:
                    cv2.rectangle(frame, 
                        (int(track_data[index][2]), int(track_data[index][3])), 
                        (int(track_data[index][2] + track_data[index][4]), int(track_data[index][3] + track_data[index][5] * upper_body_ratio)),
                        (0, 0, 255),
                        2)
                index += 1
            current_frame += 1
            out.write(frame)

        segment_num += 1

        if int(segment.character) == 1:
            with open(teacher_dataset_label_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([output_video_name, teacher_actions[int(segment.description)]])
        else:
            with open(student_dataset_label_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([output_video_name, student_actions[int(segment.description)]])
        
    cap.release()    

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
    # Attention! It's not normal IoU!
    x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3] * upper_body_ratio

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    iou = inter_area / union_area

    return iou

def binary_search(track_data, target_frame):
    low, high = 0, len(track_data) - 1

    while low <= high:
        mid = low + (high - low) // 2
        current_frame = int(track_data[mid][0])

        if current_frame == target_frame:
            high = mid - 1
        elif current_frame < target_frame:
            low = mid + 1
        else:
            high = mid - 1

    if low < len(track_data) and int(track_data[low][0]) == target_frame:
        if low != 0 and int(track_data[low-1][0]) == target_frame - 1:
            return low
        elif low == 0:
            return low
        else:
            return -1
    else:
        return -1

# Produce isolated training data

bboxes, track_data, segments = get_segments(annotation_path, track_path)
segment_id = confirm_id(segments, bboxes, track_data)
extract_target(segment_id, track_data)
# for k, v in segment_id.items():
#     attrs = vars(k)
#     print(', '.join("%s: %s" % item for item in attrs.items()))
#     print(f"{v}")
# print(len(segment_id))
# print(len(segments))
# print(len(bboxes))

# Check relevant info's validity
# for segment in segments:
#     if segment not in segment_id:
#         attrs = vars(segment)
#         print(', '.join("%s: %s" % item for item in attrs.items()))
# print(len(segments))
# print(len(track_data))
# print(len(track_data[0]))
# print(bboxes["1_hMXl6LL6"])