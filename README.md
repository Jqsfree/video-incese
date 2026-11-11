# video-incese · v0.3.0

**目标**：从原始长视频中保留同时呈现白模动作视频和渲染后视频的连续 MP4 区间。正式产物始终是原始画面 MP4，JPG 只是通过 `--evidence` 明确启用时的审计证据。

## 当前实现与限制

默认切换为 **temporal-cv**：OpenCV 识别 PiP / 左右分屏候选，在两个区域分别分析连续帧的运动、光流覆盖和场景转场，对满足双区域时序证据的连续窗口用 FFmpeg 导出 MP4。不调用 Ollama，也不需要 GPU。

**严格边界**：该模式只识别有持续动态内容的两个区域，并不能单凭运动证明“白模对应渲染视频”、区分两个碰巧同时动的教程窗口；对静止白模、低动作幅度或布局移动的场景可能漏检。结果是**时序候选**，需要用人工标注验收；不能声称真实准确率已经改善。

当前候选布局还使用固定右上 PiP / 中线分屏，不支持所有画中画位置。时间边界精度受 `--temporal-step` 和分区探测影响；每段报告里 `temporal.boundary_uncertainty_seconds` 是采样区间尺度，不是实测误差。

## Mac 更新与测试

```bash
cd video-incese
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q

video-incese \
  --input "/Users/muse/Downloads/白模/原片1/原片1.mp4" \
  --output "$HOME/Movies/video-incese-round1" \
  --content-mode temporal-cv \
  --sample-fps 2 --temporal-fps 4 --temporal-step 1
```

这条命令**不需要** `--ollama-model`，**不包含** `--dry-run`，因此会输出真正的 MP4。检测报告是 `report.json`，正式输出是 `clips/*.mp4`。如果只需要检查候选时间区间而暂不重编码，添加 `--dry-run`；如果希望看原片证据帧，添加 `--evidence`。

## 主动学习：每轮人工反馈，训练下一轮模型

主动学习不是把上轮模型的预测重新喂给自己。这里使用 **不确定性＋特征差异** 排序，优先向人索取少量有代表性的标签；人手标注才参与训练，保留报告、标签、特征版本和训练指纹。无需 LLM，使用 NumPy 实现的轻量正则化 Logistic 回归。

### 第 1 轮：生成复核队列

```bash
video-incese-learn queue \
  --report "$HOME/Movies/video-incese-round1/report.json" \
  --output "$HOME/Movies/video-incese-round1/review.csv" \
  --budget 24 \
  --preview-dir "$HOME/Movies/video-incese-round1/review-mp4"
```

`review.csv` 逐行包含时间范围、建议、概率、预览 MP4 文件路径。用表格编辑器在 `label` 列**人工填写** `keep`（保留）/ `reject`（剔除）/ `unsure`（不用于训练）。注意：每行是一个 **时间窗口** 而不是整条候选，误切边界时可逐窗反馈。

### 第 2 轮：显式训练

至少 12 个已标注窗口，且每类不少于 4 个，才允许训练，防止极少量单类样本导致失控。训练不会改变旧模型或已有视频，必须明确运行：

```bash
video-incese-learn train \
  --labels "$HOME/Movies/video-incese-round1/review.csv" \
  --output "$HOME/Movies/video-incese-model-v1.json"

video-incese \
  --input "/Users/muse/Downloads/白模/原片1/原片1.mp4" \
  --output "$HOME/Movies/video-incese-round2" \
  --content-mode temporal-cv \
  --learned-model "$HOME/Movies/video-incese-model-v1.json" \
  --learned-threshold 0.6
```

继续：`queue --report round2/report.json --output round2/review.csv`，人工校正，再执行 `train --labels round1/review.csv --labels round2/review.csv --output model-v2.json`。

训练集准确率只是内部诊断，**不能替代独立原片的时间 Precision/Recall**；不同轮次应保留固定测试片段，不用于训练。学习模型当前仅作为时序通过窗口的**额外过滤条件**，可以降低部分误收，但无法找回布局检测阶段的漏检，也不能证明语义对应关系。

## 报告与退出码

`report.json` 包含原视频信息、几何候选、`temporal_windows`（窗口时间、特征、人工审核可定位 ID、启发式分数、学习模型概率、接受状态和原因）、`temporal`（速度与失败窗口）、`clips`（实际 MP4 和状态）。解码失败窗口不会被保留；模型文件版本不兼容时明确报错，不会退回静默的旧模型。

保留旧模式：`--content-mode layout`（纯几何基线）和 `--content-mode semantic --ollama-model qwen3-vl:4b-instruct`（昂贵的本地视觉审核，非默认）。

### 注意

- `--preview-dir` 只为主动学习选中的少量窗口生成视频复核片段，不创建 JPG；`--evidence` 默认关闭。
- 输出目录每轮应独立，避免历史 MP4 残留。模型、标注与每轮报告应妥善备份；不要把用户本地视频和标签提交到公共 GitHub。
- 运行时间和误收率尚未在用户的完整真实视频上实测，不给出未经证实的性能承诺。

技术研究依据见 [研究 RFC](docs/temporal-video-segmentation-rfc.md)。
