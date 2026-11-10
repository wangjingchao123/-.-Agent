# 银发太极安全陪练

这是一个基于 Gradio、MediaPipe 和 OpenCV 的太极动作安全评估工具。当前运行入口是 `app.py`，默认读取同目录的 `taiji_24.json` 模板库。

## 目录约定

- `app.py`：当前主应用，负责视频采样、膝关节风险检测、报告和 Gradio 页面。
- `taiji_24.json`：当前使用的 24 式模板库（含左右拆分后的条目和安全规则）。
- `css/`、`js/`：旧版静态页面资源，主应用不依赖它们。
- `templates/`、`taiji_templates/`：模板数据资产。
- `tests/`：历史调试脚本和页面冒烟测试；以 `_` 开头的文件是一次性诊断脚本，不作为应用入口。
- `app_competition_backup.py`、`v03_*.py`、`taiji_*.py`：历史实验/备份实现，保留用于回溯，不要与 `app.py` 同时启动。

## 安装与运行

```powershell
python -m pip install -r requirements.txt
python app.py
```

如需使用其他模板库，可通过环境变量指定路径：

```powershell
$env:TAIJI_DATA = "upgrade_taiji.json"
python app.py
```

`TAIJI_DATA` 支持绝对路径，也支持相对于 `app.py` 所在目录的文件名。应用启动时会校验模板库必须是包含 `templates` 列表的 JSON 对象。

## 校验

提交或部署前建议先运行：

```powershell
python -m compileall -q .
```

项目中的评测方案文档是功能约束参考；其中标记为“后续可选”的设计不属于当前默认实现。
