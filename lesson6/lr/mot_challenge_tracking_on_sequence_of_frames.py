# https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md/
# python scripts/run_mot_challenge.py --BENCHMARK MOT20 --TRACKERS_TO_EVAL YOLOv8x --TRACKERS_FOLDER ./data/trackers/mot_challenge --GT_FOLDER ./data/gt/mot_challenge
# https://motchallenge.net/results/MOT20/?det=Public&orderBy=HOTA&orderStyle=ASC

from ultralytics import YOLO
import os
import cv2


'''
This script reads the MOT20 video frames and performs tracking on them using a YOLO model.
'''

def process_video(image_folder, output_file, conf_threshold=0.5):

    # Load the YOLO11 model
    model = YOLO("yolo11s.pt")

    # Get list of image paths
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()
    image_paths = [os.path.join(image_folder, img) for img in images]

    results = []
    for image_path in image_paths:

        print(image_path)
        # Read a frame from the video
        frame = cv2.imread(image_path)

        # Run YOLO11 tracking on the frame, persisting tracks between frames
        # result = model.track(frame, persist=True, device='0', conf=conf_threshold, verbose=False)  # GPU
        result = model.track(frame, persist=True, conf=conf_threshold, verbose=False)  # CPU or GPU
        results.append(result[0])

    with open(output_file, 'w') as f:
        for frame_id, result in enumerate(results, start=1):
            boxes = result.boxes
            for box in boxes:
                if box.id is None:
                    continue
                
                xyxy = box.xyxy[0].tolist()
                track_id = int(box.id.item())
                conf = box.conf.item()
                detectionClass = box.cls.item()
                
                if detectionClass != 0:
                    continue
                
                bb_left = xyxy[0]
                bb_top = xyxy[1]
                bb_width = xyxy[2] - xyxy[0]
                bb_height = xyxy[3] - xyxy[1]
                
                f.write(f'{frame_id},{track_id},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},{conf:.6f},-1,-1,-1\n')


mot20_videos = [
    'MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'
]

for video in mot20_videos:
    video_path = f'G:/MOT20/MOT20/train/{video}/img1'
    output_file = f'G:/MOT20/MOT20/train/yolo11s_131225/data/{video}.txt'
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    process_video(video_path, output_file)

print("Tracking complete. Results saved in MOTChallenge format.")