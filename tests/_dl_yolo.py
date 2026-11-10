# -*- coding: utf-8 -*-
"""断点续传下载yolov8n-pose.pt（多镜像+重试）"""
import os, time, requests

TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov8n-pose.pt")
GH = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt"
MIRRORS = [
    f"https://ghproxy.net/{GH}",
    f"https://gh-proxy.com/{GH}",
    GH,  # 直连最后试
]
TOTAL = 6528039  # 约6.5MB

for attempt in range(10):
    url = MIRRORS[attempt % len(MIRRORS)]
    have = os.path.getsize(TARGET) if os.path.exists(TARGET) else 0
    if have >= TOTAL:
        print(f"已完成: {have} bytes"); break
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        print(f"[尝试{attempt+1}] {url.split('/')[2]} 从{have}字节续传...")
        r = requests.get(url, headers=headers, timeout=30, stream=True)
        if r.status_code not in (200, 206):
            print(f"  状态码{r.status_code}，换源"); time.sleep(1); continue
        mode = "ab" if have and r.status_code == 206 else "wb"
        with open(TARGET, mode) as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk: f.write(chunk)
        size = os.path.getsize(TARGET)
        print(f"  本次下载后: {size}/{TOTAL} bytes")
        if size >= TOTAL:
            print("下载完成!"); break
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {str(e)[:80]}")
        time.sleep(2)
else:
    print("10次尝试后仍未完成")
final = os.path.getsize(TARGET) if os.path.exists(TARGET) else 0
print(f"最终文件大小: {final} bytes {'OK' if final >= TOTAL else 'INCOMPLETE'}")