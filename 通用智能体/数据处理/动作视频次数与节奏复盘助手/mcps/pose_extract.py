"""Local MediaPipe video decoding, with bounded runtime and explicit missing detections."""
import math
import os
from pathlib import Path
import sys

def extract(video,model,exercise,side,sample_fps):
    if isinstance(sample_fps,bool) or not isinstance(sample_fps,(int,float)) or not math.isfinite(sample_fps) or not 1<=sample_fps<=15:
        raise ValueError('sample_fps must be a finite number from 1 to 15')
    dependencies=os.environ.get('FITNESS_VIDEO_DEPENDENCIES')
    if dependencies and dependencies not in sys.path: sys.path.insert(0,dependencies)
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError: raise ValueError('video dependencies unavailable; install requirements-video.txt and configure model')
    if exercise not in ('squat','curl') or side not in ('left','right'): raise ValueError('supported exercise: squat/curl; side: left/right')
    model=Path(model)
    if not model.is_file(): raise ValueError('pose model not configured; set FITNESS_POSE_MODEL')
    capture=cv2.VideoCapture(str(video))
    if not capture.isOpened(): raise ValueError('video could not be opened')
    fps=capture.get(cv2.CAP_PROP_FPS); frame_count=capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if not math.isfinite(fps) or not 0<fps<=240:
        capture.release(); raise ValueError('video frame rate invalid')
    if frame_count/fps>120:
        capture.release(); raise ValueError('video duration exceeds 120 seconds')
    options=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model)),running_mode=vision.RunningMode.VIDEO,num_poses=1,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
    indices=(23,25,27) if exercise=='squat' else (11,13,15)
    if side=='right': indices=tuple(index+1 for index in indices)
    results=[]; position=0; next_sample=0
    try:
        with vision.PoseLandmarker.create_from_options(options) as detector:
            while True:
                valid,frame=capture.read()
                if not valid: break
                seconds=position/fps; position+=1
                if seconds>120: raise ValueError('decoded video exceeds 120 seconds')
                if seconds+1e-6<next_sample: continue
                next_sample=seconds+1/sample_fps
                height,width=frame.shape[:2]
                if max(height,width)>960:
                    scale=960/max(height,width); frame=cv2.resize(frame,(round(width*scale),round(height*scale))); height,width=frame.shape[:2]
                converted=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=converted),round(seconds*1000))
                points=[]
                if result.pose_landmarks:
                    for index in indices:
                        landmark=result.pose_landmarks[0][index]
                        points.append({'x':landmark.x,'y':landmark.y,'visibility':landmark.visibility})
                results.append({'time_seconds':round(seconds,6),'width':width,'height':height,'points':points})
    finally: capture.release()
    if not results: raise ValueError('video had no decodable frames')
    return results,{'engine':'MediaPipe Pose Landmarker','model':model.name,'sample_fps':sample_fps,'container_fps':fps,'decoded_frames':position,'timestamp_basis':'decoded frame index / container FPS','sampled_frames':len(results)}
