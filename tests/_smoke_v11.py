# -*- coding: utf-8 -*-
"""v11冒烟测试：funasr语音+YOLO姿态+高危统计+复盘报告"""
import sys, os, threading
import numpy as np

# 阻止模块级模型加载线程启动（避免与运行中服务器并发下载冲突）
_orig_start = threading.Thread.start
def _selective_start(self):
    t = getattr(self, '_target', None)
    if t is not None and getattr(t, '__name__', '') in ('_load_asr_model', '_load_yolo_model'):
        return  # 跳过模型加载
    _orig_start(self)
threading.Thread.start = _selective_start

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import taiji_coach as tc

threading.Thread.start = _orig_start  # 恢复

PASS = 0
FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  [PASS] {name}")
    else:
        FAIL += 1; print(f"  [FAIL] {name} {detail}")

print("=== 1. check_risk 高危姿态判断（YOLO 17关键点） ===")
def mk_kpts(overrides=None):
    """正常站姿：鼻(320,80) 髋L(300,240)R(340,240) 膝L(300,330)R(340,330) 踝L(300,420)R(340,420)"""
    k = np.zeros((17, 3), dtype=np.float32)
    base = {0:(320,80), 11:(300,240), 12:(340,240), 13:(300,330), 14:(340,330), 15:(300,420), 16:(340,420)}
    for i,(x,y) in base.items(): k[i] = (x,y,0.9)
    if overrides:
        for i,(x,y) in overrides.items(): k[i] = (x,y,0.9)
    return k

r = tc.check_risk(mk_kpts())
check("正常站姿无风险", r == [], f"got {r}")
r = tc.check_risk(mk_kpts({13:(280,330)}))  # 左膝x280 < 左踝300-10
check("左膝内扣", "knee_in" in r, f"got {r}")
r = tc.check_risk(mk_kpts({14:(360,330)}))  # 右膝x360 > 右踝340+10
check("右膝内扣", "knee_in" in r, f"got {r}")
r = tc.check_risk(mk_kpts({0:(320,220)}))   # 鼻y220 > 髋中心240-30=210
check("弯腰塌腰", "bend_back" in r, f"got {r}")
r = tc.check_risk(mk_kpts({11:(300,210), 12:(340,250)}))  # 髋y差40>25
check("重心歪斜", "tilt_body" in r, f"got {r}")
r = tc.check_risk(np.zeros((10,3)))
check("不足17点返回空", r == [], f"got {r}")
r = tc.check_risk(mk_kpts({0:(320,220), 13:(280,330), 11:(300,200), 12:(340,260)}))
check("三风险同时", len(r)==3 and set(r)=={"knee_in","bend_back","tilt_body"}, f"got {r}")
# 边界：鼻y=210恰好不触发弯腰（>210才触发）
r = tc.check_risk(mk_kpts({0:(320,210)}))
check("弯腰边界值210不触发", "bend_back" not in r, f"got {r}")

print("=== 2. get_camera_frame 无摄像头降级 ===")
tc.frame_shared = None
img = tc.get_camera_frame()
check("占位黑图480x640x3", img is not None and img.shape==(480,640,3), f"{img.shape if img is not None else None}")
tc.frame_shared = np.full((480,640,3), 128, dtype=np.uint8)
img = tc.get_camera_frame()
check("有帧返回帧", img is not None and int(img[0,0,0])==128)

print("=== 3. audio_recognize 降级路径 ===")
tc.ASR_ENABLE = False
r = tc.audio_recognize((48000, np.zeros(1000, dtype=np.float32)))
check("语音关闭降级", "语音关闭" in r, f"got {r}")
tc.ASR_ENABLE = True
r = tc.audio_recognize(None)
check("audio为None降级", isinstance(r, str), f"got {r}")
tc.asr_status = "loading"; tc.asr_model = None
r = tc.audio_recognize((48000, np.zeros(1000, dtype=np.float32)))
check("模型加载中降级", "加载中" in r, f"got {r}")
tc.asr_status = "failed"; tc.asr_model = None
r = tc.audio_recognize((48000, np.zeros(1000, dtype=np.float32)))
check("模型不可用降级", "不可用" in r or "手动" in r, f"got {r}")

print("=== 4. update_action_table 表格编辑 ===")
tc.action_dict = {"起势":8, "云手":6}; tc.current_action_name = "起势"
dd, msg = tc.update_action_table([])
check("空表返回2值", isinstance(dd, object) and isinstance(msg, str))
check("空表保留原列表", "不能为空" in msg, f"got {msg}")
dd, msg = tc.update_action_table([["白鹤亮翅",9],["单鞭",5]])
check("正常更新字典", "白鹤亮翅" in tc.action_dict and tc.action_dict["白鹤亮翅"]==9, f"{tc.action_dict}")
check("更新后记录清空", all(v==0 for rd in tc.practice_record.values() for v in rd.values()))
dd, msg = tc.update_action_table([["野马分鬃","abc"],["",5],["云手",6]])
check("非法行跳过", "云手" in tc.action_dict and "野马分鬃" not in tc.action_dict, f"{tc.action_dict}")
tc.current_action_name = "云手"
dd, msg = tc.update_action_table([["起势",8]])
check("当前动作被删回退首个", tc.current_action_name=="起势", f"got {tc.current_action_name}")

print("=== 5. manual_switch / switch_asr ===")
r = tc.manual_switch("起势")
check("切换存在动作", "起势" in r and tc.current_action_name=="起势", f"got {r}")
r = tc.manual_switch("不存在")
check("不存在动作提示", "不在列表" in r, f"got {r}")
r = tc.switch_asr(True)
check("开启语音", "开启" in r, f"got {r}")
r = tc.switch_asr(False)
check("关闭语音", "关闭" in r, f"got {r}")
tc.ASR_ENABLE = True

print("=== 6. generate_report 复盘报告 ===")
tc.practice_record = {"起势": {"knee_in":0,"bend_back":0,"tilt_body":0}}
r = tc.generate_report()
check("全0无高危", "没有检测到高危" in r, f"got {r[:60]}")
tc.practice_record = {
    "起势": {"knee_in":5,"bend_back":2,"tilt_body":0},
    "云手": {"knee_in":1,"bend_back":8,"tilt_body":3},
}
r = tc.generate_report()
check("含重点标记", "重点标记" in r, f"got {r[:80]}")
check("标红最高频弯腰8次", "弯腰塌腰" in r and "8次" in r, f"got {r[-120:]}")

print("=== 7. reset_record ===")
tc.practice_record = {"起势": {"knee_in":5,"bend_back":2,"tilt_body":0}}
r1, r2 = tc.reset_record()
check("返回2值", isinstance(r1,str) and isinstance(r2,str))
check("记录清零", all(v==0 for rd in tc.practice_record.values() for v in rd.values()))
check("清空提示", "清空" in r1, f"got {r1}")

print("=== 8. 模块级状态 ===")
check("asr_status有效", tc.asr_status in ("loading","ready","failed"))
check("yolo_status有效", tc.yolo_status in ("loading","ready","failed"))
check("记录结构3键", all(set(v.keys())=={"knee_in","bend_back","tilt_body"} for v in tc.practice_record.values()))
check("action_dict非空", len(tc.action_dict) > 0)

print(f"\n{'='*40}")
print(f"v11冒烟测试：{PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL else 0)