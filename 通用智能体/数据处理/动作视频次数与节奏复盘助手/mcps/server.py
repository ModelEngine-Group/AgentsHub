from runtime import *

from pathlib import Path
import importlib.util
import math
import os

def analyze(data):
    mode=data.get('mode','keypoints'); exercise=data.get('exercise','squat'); side=data.get('side','left')
    if exercise not in ('squat','curl') or side not in ('left','right'): raise ValueError('supported exercise: squat/curl; side: left/right')
    low=number(data.get('low_angle',100),'low angle',1,178); high=number(data.get('high_angle',160),'high angle',2,179)
    if low>=high: raise ValueError('angle thresholds inverted')
    phase=number(data.get('min_phase_seconds',.2),'minimum phase seconds',0,5); gap=number(data.get('max_gap_seconds',.6),'max gap seconds',.05,5); confidence=number(data.get('minimum_visibility',.5),'visibility',0,1)
    if mode=='video':
        file=runtime.input_file(data.get('video_filename')); model=os.environ.get('FITNESS_POSE_MODEL','')
        if file.suffix.lower() not in ('.mp4','.mov','.avi','.webm'): raise ValueError('unsupported video extension')
        spec=importlib.util.spec_from_file_location('fitness_pose_extract',Path(__file__).with_name('pose_extract.py')); extractor=importlib.util.module_from_spec(spec); spec.loader.exec_module(extractor)
        frames,source=extractor.extract(file,model,exercise,side,number(data.get('sample_fps',5),'sample fps',1,15))
        source['mode']='video'
    elif mode=='keypoints': frames=rows(data.get('frames'),'frames',10000); source={'mode':'keypoint_fixture','note':'使用者提供的关键点样例，不是实际视频识别结果。'}
    else: raise ValueError('mode must be video or keypoints')
    cycles=[]; diagnostics=[]; previous=None; state='need_high'; held_label=None; held_start=None; cycle_start=None; down_time=None
    for frame in frames:
        obj(frame); stamp=number(frame.get('time_seconds'),'frame time',0,120)
        if previous is not None and stamp<=previous: raise ValueError('frame timestamps must strictly increase')
        if previous is not None and stamp-previous>gap:
            state='need_high'; held_label=None; held_start=None; cycle_start=None; down_time=None
        previous=stamp; points=rows(frame.get('points',[]),'points',3); angle=None
        width=number(frame.get('width',1),'width',1,10000); height=number(frame.get('height',1),'height',1,10000)
        if len(points)==3:
            for point in points:
                obj(point); number(point.get('x'),'x',-2,3); number(point.get('y'),'y',-2,3); number(point.get('visibility'),'visibility',0,1)
            if all(point['visibility']>=confidence for point in points):
                a,b,c=points; ax=(a['x']-b['x'])*width; ay=(a['y']-b['y'])*height; bx=(c['x']-b['x'])*width; by=(c['y']-b['y'])*height
                denominator=math.hypot(ax,ay)*math.hypot(bx,by)
                if denominator>1e-12: angle=math.degrees(math.acos(max(-1,min(1,(ax*bx+ay*by)/denominator))))
        diagnostics.append({'time_seconds':stamp,'angle_degrees':round(angle,4) if angle is not None else None})
        if angle is None:
            state='need_high'; held_label=None; held_start=None; cycle_start=None; down_time=None; continue
        label='high' if angle>=high else 'low' if angle<=low else 'middle'
        if held_label!=label: held_label=label; held_start=stamp
        if stamp-held_start+1e-9<phase: continue
        if label=='high':
            if state=='need_high': state='need_low'; cycle_start=held_start
            elif state=='need_return':
                cycles.append({'start_seconds':cycle_start,'low_seconds':down_time,'end_seconds':stamp,'cycle_seconds':round(stamp-cycle_start,6)})
                state='need_low'; cycle_start=held_start; down_time=None
        elif label=='low' and state=='need_low': state='need_return'; down_time=held_start
    values=[row['cycle_seconds'] for row in cycles]
    return {'source':source,'exercise':exercise,'side':side,'rep_count':len(cycles),'cycles':cycles,'mean_cycle_seconds':round(sum(values)/len(values),6) if values else None,'sample_count':len(frames),'valid_angle_samples':sum(row['angle_degrees'] is not None for row in diagnostics),'thresholds':{'low_angle':low,'high_angle':high,'minimum_phase_seconds':phase,'maximum_gap_seconds':gap,'minimum_visibility':confidence},'note':'只统计所选侧的二维角度周期；阈值是软件计数参数，遮挡或缺点会中断周期，不评价动作标准、伤病或康复。','_artifacts':[('movement-cycles.csv',csv_text(cycles,['start_seconds','low_seconds','end_seconds','cycle_seconds']),'text/csv'),('angle-timeline.csv',csv_text(diagnostics,['time_seconds','angle_degrees']),'text/csv')]}

runtime = Runtime('fitness-video-rhythm', '动作视频次数与节奏复盘助手', 'fitness_video_rhythm', '本地提取视频姿态点或读取明确标注的关键点样例，按角度滞回与持续时间统计完整动作周期。', package_root=Path(__file__).resolve().parents[1])

def run(payload: str) -> str:
    return runtime.process(payload, analyze)

if __name__ == "__main__":
    runtime.main(run)
