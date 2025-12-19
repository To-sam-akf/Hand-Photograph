# Hand_Photograph

中文说明 — 本项目包含用于手部/手指检测与基于颜色与YOLO模型的识别脚本，演示如何从摄像头或图片中检测手部并做后续处理（例如手指检测、颜色检测等）。

## 目录（关键文件）
- [main.py](https://github.com/To-sam-akf/Hand-Photograph/blob/bbd0728de9a176ef6d51de74e9dffb857324dd17/Hand_Photograph/main.py) — 项目入口脚本，负责启动摄像头/读取图片并调用其它模块进行处理。
- [yolo.py](https://github.com/To-sam-akf/Hand-Photograph/blob/bbd0728de9a176ef6d51de74e9dffb857324dd17/Hand_Photograph/yolo.py) — 封装了基于 YOLO 的目标检测相关逻辑（加载模型、推理、后处理等）。
- [colorDetection.py](https://github.com/To-sam-akf/Hand-Photograph/blob/bbd0728de9a176ef6d51de74e9dffb857324dd17/Hand_Photograph/colorDetection.py) — 基于颜色空间（HSV 等）进行的颜色检测，用于分割或筛选出特定颜色区域（可用于指示环或标记等）。
- [fingerDetection.py](https://github.com/To-sam-akf/Hand-Photograph/blob/bbd0728de9a176ef6d51de74e9dffb857324dd17/Hand_Photograph/fingerDetection.py) — 基于轮廓与几何的手指检测/计数逻辑（在手部分割后运行）。
- [models/](https://github.com/To-sam-akf/Hand-Photograph/tree/bbd0728de9a176ef6d51de74e9dffb857324dd17/Hand_Photograph/models) — 用于存放预训练模型文件（如 YOLO 权重、配置等）。
- __pycache__/ — Python 缓存（无需提交/关注）。

## 功能概述
- 从摄像头或静态图像中检测手部目标（基于 YOLO）；
- 对检测到的手部区域进行颜色分割（例如基于 HSV 范围），用于强调或筛选特定区域；
- 在分割后的手部图像上进行手指检测/计数；
- 可组合使用：先用 YOLO 检测手，再在手部 ROI 上使用颜色检测或指尖检测提高鲁棒性。

## 环境与依赖（建议）
- Python 3.8+
- 必要的常用库（示例）：
  - opencv-python
  - numpy
  - torch（若使用 PyTorch 权重/YOLO 实现）
  - torchvision
  - imutils（可选）
  - matplotlib（可选，用于可视化）
- 安装示例：
```bash
python -m pip install --upgrade pip
python -m pip install opencv-python numpy torch torchvision imutils
```
> 注：具体依赖请根据 `yolo.py` / `main.py` 中的 import 来调整。如果使用特定的 YOLO 发行版（如 ultralytics/yolov5），请按该项目说明安装对应依赖。

## 快速开始（示例）
1. 将预训练模型放到 `Hand_Photograph/models/` 目录（例如 yolov5s.pt 或其它权重；具体文件名参见 yolo.py 中加载路径）。
2. 启动摄像头实时检测：
```bash
python Hand_Photograph/main.py
```
3. 使用静态图片（如果 main.py 支持参数，可按脚本接受的参数修改）：
```bash
python Hand_Photograph/main.py --source path/to/image.jpg
```
（如果 main.py 不带参数，请打开文件查看如何传入图像路径或修改为合适的调用方式。）

## 配置指南
- 颜色检测（colorDetection.py）
  - 通常使用 HSV 下限/上限来筛选颜色。打开 `colorDetection.py`，找到 HSV 范围（lower/upper）并根据环境光调整。
- YOLO 设置（yolo.py）
  - 若需更换模型或调整置信度阈值、NMS 阈值，请查看 `yolo.py` 中的相应变量或函数参数并调整。
- 手指检测（fingerDetection.py）
  - 可能包含轮廓面积阈值、凸缺陷阈值等参数，可根据摄像头分辨率或手部大小微调。

## 常见问题与排查
- “无法加载模型/权重”：
  - 确认模型文件已放入 `models/` 并且 `yolo.py` 中的路径与文件名匹配。
  - 如果使用 PyTorch 权重，确认 PyTorch 版本与权重兼容。
- “检测不稳定/误检”：
  - 调整 YOLO 的置信度阈值或 NMS 参数；
  - 在 colorDetection 中调整 HSV 范围以适应当前光照；
  - 在 fingerDetection 中调整轮廓过滤阈值。
- “摄像头打不开”：
  - 确认摄像头索引（0、1 等），或将 main.py 的 source 改为视频文件路径进行测试。

## 如何贡献
- 欢迎提交 issue 说明问题或 feature request；
- 欢迎 fork 后提交 PR，包括但不限于：
  - 增加 requirements.txt 或 environment.yml；
  - 在 main.py 中添加更友好的 CLI（例如使用 argparse）；
  - 提升检测精度或增加实时性能优化（如采用更快的模型、使用 ONNX/TensorRT 加速等）。

## 开发者/作者
- 仓库：To-sam-akf/Hand-Photograph
- 联系：请通过 GitHub issue 或者 Pull Request 进行交流。
