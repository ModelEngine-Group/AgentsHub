# 动作视频次数与节奏复盘助手知识库

## 识别来源 VIDEO-SOURCE

视频模式使用本地 MediaPipe Pose Landmarker Lite 提取所选侧髋膝踝或肩肘腕关键点，按实际宽高计算二维夹角。官方说明 https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker?tab=Python 。模型单独下载，遵守其许可证。

## 完整周期 VIDEO-CYCLE

计数需要高角度、低角度、返回高角度三阶段，并满足可配置保持时长。丢失关键点、可见度不足或过长采样间隔会重置。计数阈值不是医学或动作合格标准。
