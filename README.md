# video-incese

在连续视频中定位并截取 **参考模型/原始动作与生成视频同时出现** 的对比片段，支持右上角画中画（PiP）和左右分屏。专门为用户提供的四段样本布局设定默认参数。不会修改输入文件。

> **检测边界：** 默认 OpenCV 仅证明存在目标布局，不能单独证明画面内容确实是对应的模型与生成结果。`--ollama-model` 提供实验性的本地视觉语义复核（仅验证每个候选片段中间的一帧），仍须人工抽检。不提供任何未经测试的生产准确率承诺。

## macOS (Apple Silicon / Intel) 部署

安装 Homebrew（若已安装则跳过），然后执行：

```bash
brew install python@3.12 ffmpeg git
python3.12 --version
ffmpeg -version
ffprobe -version

git clone https://github.com/Jqsfree/video-incese.git
cd video-incese
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
video-incese --help
```

默认只用 CPU 上的 OpenCV，无 CUDA、PyTorch 或云端 API 依赖。M 系列 Mac 可运行。若 `brew` 尚未安装，参考 https://brew.sh/ 的官方说明。

## 先预览检测，再截取

```bash
# 路径可包含空格和中文；dry-run 不导出片段，只生成 JSON 报告
video-incese --input "$HOME/Movies/source.mp4" --output "$HOME/Movies/incese-out" --dry-run

# 真正输出：clips/*.mp4 + evidence/*.jpg + report.json
video-incese --input "$HOME/Movies/source.mp4" --output "$HOME/Movies/incese-out" --evidence

# 更密集采样减少边界误差，CPU 时间会增加
video-incese --input "$HOME/Movies/source.mp4" --output "$HOME/Movies/incese-out-4fps" --sample-fps 4 --evidence
```

**建议先查看 `report.json` 的候选时间范围，再逐段检查 `clips/`。** 时间切分的误差通常取决于采样间隔，导出阶段使用重新编码而非关键帧复制，避免非关键帧边界导致整段错位。

可调整参数：`--layout auto|pip|split`、`--pip-x 0.655`、`--pip-y 0.338`（画中画左上角坐标/视频尺寸）、`--pip-vertical 4`、`--pip-horizontal 3`（越大越严格）、`--split-threshold 26`、`--merge-gap 2.6`（秒）、`--min-duration 0.8`（秒）。较长缺口（不超过 `--merge-gap`）只有在期间持续存在弱画中画边界证据时才合并；完全没有边界证据时不合并，避免把真正不含对比画面的片段混入。

目前只支持顶部右侧小窗和中线附近左右分屏，其他布局请先用参数标定或扩展算法。

## 可选：本地视觉模型复核（实验性）

```bash
# 已安装 Ollama 时（没有则按 https://ollama.com/ 获取 macOS 版本）
ollama pull qwen3-vl:4b
# 确认 Ollama 本地服务已启动，模型可用
ollama list
video-incese --input "$HOME/Movies/source.mp4" --output "$HOME/Movies/incese-vlm" \
  --ollama-model qwen3-vl:4b --evidence
```

仅将候选片段的中间一帧提交至 `127.0.0.1:11434`。当 Ollama 不可用、模型缺失或输出无法解析时，该候选片段会记录为 `verification_error`（**不导出**），其余片段继续处理；连续 3 次失败后剩余片段标记为 `verification_skipped`。可处理完采样和片段时会写出 `report.json`；前置参数或媒体探测失败只输出 stderr。VLM 实际语义准确率仍需人工标注验证。

## 输出契约

```text
incese-out/
  clips/               # 每个有效连续时间段输出一个可播放 MP4（无命中则不创建）
  evidence/            # --evidence 时每个候选一个中间帧；被复核拒绝的带 _rejected 后缀
  report.json          # 输入信息、参数、sampling、warnings、采样分数、片段状态
```

文件名包含原始 stem 和精确起止时间；默认覆盖相同文件名输出，建议不同参数使用独立输出目录以方便对比。原始视频不会写入。运行失败时检查 `stderr`；常见原因是 FFmpeg 缺失、编解码器不可用或无权写入输出目录。

## 退出码与失败记录

片段级错误会记入 `report.json`；输入参数、媒体探测及读取初始化等前置失败只输出 stderr。

| 情况 | 记录位置 | 退出码 |
|---|---|---|
| 参数或前置环境错误 | stderr | 1 |
| `frame_error` / `verification_error` / `verification_skipped` / `export_error` | `clips[].status`，stderr | 1 |
| 孤立采样点不可解码，跳过并继续 | `sampling.failed_times`、`warnings` | 0 |
| 末尾未分析超过 2 秒（或一个采样间隔） | `sampling.incomplete=true` | 2 |

`rejected_semantic` 表示 VLM 拒绝候选，不属于程序故障；`evidence_error` 目前记入对应片段，但不会导致退出码非零。**本补丁是可靠性修复，不提升内容筛选精度，网页/软件操作画面误收仍然存在。**

## 验证要求与当前进度

- `pytest`：合成 PiP/分屏/纯单画面与时间区间合并测试，以及解码失败、VLM 异常、导出失败与拒绝证据测试。CI 在 macOS、Ubuntu 各两个 Python 版本上执行。
- 样本：四段上传样本用于开发回归，**都为正例**；不能用于证明真实负例精度。未经足够多真实正负例标注，不报告 Precision、Recall 或 F1。
- 人工验收：每条原始长视频人工标记起止点，比较脚本输出的起止点、误收和漏收；检查声音同步和 MP4 可播放。

## 已知限制

目前是基于 **画面几何结构** 的候选片段提取器。缺少视觉语义判断时，其他类型的画中画（游戏直播、人像主播、新闻播报）可能被保留；相同场景的低对比度小窗可能被漏检。`--ollama-model` 仅单帧语义核验，不保证完整片段每一帧都满足要求。对负例准确率和召回率应在用户本地真实长视频上评估后再考虑生产使用。
