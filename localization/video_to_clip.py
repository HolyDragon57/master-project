import cv2
import os
import json
import os.path as osp

def slicing(track_result, track_video, output_dir):
    cap = cv2.VideoCapture(track_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    with open('./config.json', 'r') as f:
        config = json.load(f)
    clip_frames = int(config['localization']['clip_duration'] * fps)
    step_frames = int(config['localization']['time_step'] * fps)
    
    clip_number = 0
    frame_id = 0
    frame_count = 1
    track_data = []
    index = []
    with open(track_result, 'r', encoding='utf-8') as track_file:
        for line in track_file:
            numbers = [num for num in line.strip().split(",")]
            track_data.append(numbers[:-3])
            if int(numbers[0]) != frame_id:
                frame_id = int(numbers[0])
                if frame_count % step_frames == 0:
                    index.append(len(track_data) - 1)
                if frame_count == clip_frames:
                    tracklets = division(track_data[:-1])
                    with open(os.path.join(output_dir, 'clips.txt'), 'a') as f:
                        for k, v in tracklets.items():
                            for line in v:
                                line_str = ",".join(num for num in line)
                                f.write(line_str + "\n")
                            f.write("\n")
                    # with open('./test_output.txt', 'a') as f:
                    #     for line in track_data[:-1]:
                    #         line_str = ",".join(num for num in line)
                    #         f.write(line_str + "\n")
                    #     f.write("\n")
                    track_data = track_data[index[0]:]
                    index[1] -= index[0]
                    index = index[1:]
                    clip_number += 1
                    frame_count = clip_frames - step_frames
                frame_count += 1
    if len(track_data) != 0:
        tracklets = division(track_data)
        with open(os.path.join(output_dir, 'clips.txt'), 'a') as f:
            for k, v in tracklets.items():
                for line in v:
                    line_str = ",".join(num for num in line)
                    f.write(line_str + "\n")
                f.write("\n")
    # print(track_data[0])
    # print(len(track_data))
    cap.release()
    return output_dir + '/clips.txt'


def division(track_data):
    tracklets = {}
    for line in track_data:
        if int(line[1]) not in tracklets:
            tracklets[int(line[1])] = []
        tracklets[int(line[1])].append(line)
    return tracklets
    # for k, v in tracklets.items():
    #     print(k)
    #     for item in v:
    #         print(item)
    #     print()



def split_video(video_path, output_path, clip_duration=4, step=2):
    
    video_capture = cv2.VideoCapture(video_path)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    clip_frames = int(clip_duration * fps)
    overlap_frames = int((clip_duration - step) * fps)

    os.makedirs(output_path, exist_ok=True)

    clip = []

    for _ in range(clip_frames):
        ret, frame = video_capture.read()
        if not ret:
            break
        clip.append(frame)

    clip_number = 0
    current_frame = clip_frames

    while current_frame < total_frames:

        clip_output_path = os.path.join(output_path, f"{osp.splitext(osp.basename(video_path))[0]}_clip_{clip_number}.mp4")
        print(f"Writing {output_path}/{osp.splitext(osp.basename(video_path))[0]}_clip_{clip_number}.mp4")
        clip_video_writer = cv2.VideoWriter(clip_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(video_capture.get(3)), int(video_capture.get(4))))

        for frame in clip:
            clip_video_writer.write(frame)
        clip_video_writer.release()

        clip = clip[clip_frames - overlap_frames:]

        for _ in range(clip_frames - overlap_frames):
            ret, frame = video_capture.read()
            if not ret:
                break
            clip.append(frame)
        
        current_frame += clip_frames - overlap_frames
        clip_number += 1

        if current_frame >= total_frames:
            break

    video_capture.release()


