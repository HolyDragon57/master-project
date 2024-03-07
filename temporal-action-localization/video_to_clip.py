import cv2
import os
import os.path as osp

def split_video(video_path, output_path, clip_duration=3, overlap_duration=2):
    
    video_capture = cv2.VideoCapture(video_path)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    clip_frames = int(clip_duration * fps)
    overlap_frames = int(overlap_duration * fps)

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

if __name__ == "__main__":
    video_path = "../demo.mp4" 
    output_path = "output_clips"  

    split_video(video_path, output_path)
