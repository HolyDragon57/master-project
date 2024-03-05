import cv2
import os

def split_video(video_path, output_path):
    
    video_capture = cv2.VideoCapture(video_path)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    clip_duration = 3
    frames_per_clip = int(fps * clip_duration)

    os.makedirs(output_path, exist_ok=True)

    clip_number = 0
    current_frame = 0

    while current_frame < total_frames:

        clip_output_path = os.path.join(output_path, f"{video_path}_clip_{clip_number}.mp4")
        print(f"Writing {output_path}/{video_path}_clip_{clip_number}.mp4")
        clip_video_writer = cv2.VideoWriter(clip_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(video_capture.get(3)), int(video_capture.get(4))))

        for _ in range(frames_per_clip):
            ret, frame = video_capture.read()
            if not ret:
                break
            clip_video_writer.write(frame)
            current_frame += 1

            if current_frame >= total_frames:
                break

        clip_video_writer.release()
        clip_number += 1

    video_capture.release()

if __name__ == "__main__":
    video_path = "demo.mp4" 
    output_path = "output_clips"  

    split_video(video_path, output_path)
