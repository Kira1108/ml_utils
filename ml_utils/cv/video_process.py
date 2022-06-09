import cv2
import os
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

def get_now():
    return int(datetime.now().timestamp())

def extract_video(video_path:str, dest:str = "./videodata", file_prefix = "frame", kps = 1):
    # extract basename of a vidoe file
    base_name = os.path.basename(video_path).split(".")[0]
    
    # make a folder that stores frame results
    video_target_path = os.path.join(dest, base_name + str(get_now()))
    os.makedirs(video_target_path,exist_ok = True)
    
    # capture video frames 1 frame at a time
    vidcap = cv2.VideoCapture(video_path)
    
    # frame per seconds
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    
    # skip frames hops
    hop = round(fps / kps)
    
    count = 1
    success = 1
    while success:
        success,image = vidcap.read()
        if success and (count % hop == 0):
            filepath = os.path.join(video_target_path, f"{file_prefix}_{count}.jpg")
            cv2.imwrite(filepath, image)    
        count += 1
        logger.info(f"Extract {count} frames")
        
    return video_target_path