"""用真实上传视频验证完整分析链路：VideoCapture→MediaPipe→DTW→评分"""
import sys
sys.path.insert(0, ".")
import v03_gradio_final_verified as m

video = "uploads/10月29日 (1)(16)-1.mp4"
action = "起势"
print(f"视频：{video}\n招式：{action}")
rep, tip, score, dist, _ = m.analyze_video_file(video, action)
print(rep)
print(f"\n[返回值] score={score}, dist={dist}")