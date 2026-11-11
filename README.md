# video-incese · v0.3.2

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

## 将人工剪辑直接导入正例（v0.3.1）

已有人工确认的纯正例 MP4 时，不必把它重新人工拆成 1 秒标签。使用
`video-incese-learn import-positive`，以原视频每秒的时序证据生成报告和
人工正例标签；不调用 Ollama。为防止剪辑边缘污染和布局误识别，边缘窗口或
没有检测到布局的窗口标记为 `unsure`，**不是强行标成负例**。程序也会统计
“人工保留但当前时序规则拒绝”的窗口，帮助诊断漏检。

```bash
video-incese-learn import-positive \
  --clip "/path/to/manual-1.mp4" \
  --clip "/path/to/manual-2.mp4" \
  --clip "/path/to/manual-3.mp4" \
  --clip "/path/to/manual-4.mp4" \
  --confirmed-reviewed \
  --reports-dir "$HOME/Movies/video-incese-gold/reports" \
  --output "$HOME/Movies/video-incese-gold/positive-labels.csv"
```

注意：

- `--confirmed-reviewed` 明确表示四段内容已经由人确认属于应保留的视频；
  禁止将程序自己切出的片段未经复核当作正例导入。
- `positive-labels.csv` 的 `keep` 是剪辑级人工标签向内部窗口的映射。
  存在慢动作/人物停顿的窗口也可能被标记为 keep，正是需要学习的正例。
- 这些片段 **只有正例没有负例**；要真正训练第二轮，需要用完整原片的
  `video-incese-learn queue` 选择负例并人工填写 `reject`。然后执行：
  `video-incese-learn train --labels positive-labels.csv --labels round1/review.csv --output model-v2.json`。
- 短视频之间的相邻窗口高度相关。不可随机按窗口划分训练集/测试集；
  最好按原片分组，至少把其中一条完整原片留作从未参与训练的独立验证集。
- 未提供原片时无法将这些剪辑的起止时间映射回长视频，无法计算长视频整体召回率。
- 保持原始人工剪辑 MP4 不变；只生成标注 CSV 与报告 JSON，避免上传私人视频到公共 GitHub。

## v0.3.2：真实人工正例报告反馈后的改动

四段用户人工保留的视频在 v0.3.1 报告中合计 70 个 1 秒左右的窗口：
61 个通过，7 个运动/布局不足被拒，2 个末端解码失败。
部分片段的参考模型短暂停顿、局部布局边缘不明显，因此 `dual_motion` 的逐窗硬规则造成错误截断。

- 时间上下文：默认 `--temporal-context-gap 2`，仅在附近有通过窗口、
  两区域存在图像结构和生成区域运动、无硬错误/硬场景切换的条件下，恢复
  ≤2 秒的内部不确定区间，或 ≤1.5 秒的末尾片段；设置 `0` 可关闭。
  **不得**据此推断两个视频语义对应；`context_motion_pause`、
  `context_layout_recovered` 在报告中单独标注，优先人工复核。
- 文件尾：FFprobe 的 container duration 可能晚于最后可解码帧的时间戳；
  只允许在真实文件尾做有限的向前帧重试并记录 `recovered_tail_samples`。
  中间位置解码失败依旧是明确错误。
- 已训练的学习模型：不再仅作为额外否决器。当概率足够高并且存在最小
  视觉时序证据时，可恢复由手工阈值误拒的窗口，记录
  `accepted_learned`。没有人标负例时仍**禁止**训练。
- 报告同时保留 `raw_accepted` / `raw_reason` 与最终的 `accepted` /
  `reason`，便于判断每轮改进是否以放宽规则换取漏收减少。

以上只基于四段已知正例发现缺陷，不能证明长原片的误收率改善；请将
第一轮原片分段与人剪结果对比，并优先查看上下文恢复的片段。 
