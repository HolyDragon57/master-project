import os
import subprocess
import json
from localization.video_to_clip import slicing
from reannotation.vis import aggregation

with open('./config.json', 'r') as f:
    config = json.load(f)
output_path = config['path']['output_path']
input_path = config['path']['input_path']

def check_label_file(folder_path, file_name):

    txt_files = [file for file in os.listdir(folder_path) if file.startswith(file_name) and file.endswith(".txt")]
    assert len(txt_files) == 1, "There should be exactly one tracking txt file for this video."
    return txt_files[0]


if __name__ == "__main__":
    processed_videos = set()
    processed_videos.update(os.listdir(output_path))

    for file in os.listdir(input_path):
        file_name = os.path.splitext(file)[0]
        # if file_name not in processed_videos:
        output_dir =  os.path.join(output_path, file_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Human Tracking
        # track_command = f"python ./human-tracking/src/main.py video -f ./human-tracking/src/yolox_x_mix_det.py -c ./human-tracking/pretrained/bytetrack_x_mot17.pth.tar --input_path ./input/{file} --output_path {output_dir}"
        # print("Executing human tracking command...")
        # print(track_command)
        # subprocess.run(track_command, shell=True)
        # print(f"Storing tracking result in ./output/{file_name}")

        # Temporal & spatial localization
        # track_result = check_label_file(output_dir, file_name)
        # track_video = os.path.join(output_dir, os.path.splitext(track_result)[0], file)
        # clips_path = slicing(os.path.join(output_dir, track_result), track_video, output_dir)

        # aggregation(clips_path)
        aggregation('./output/demo/clips.txt')
        # Action Recognition

        # Reannotation
