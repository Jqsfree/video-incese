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

可调整参数：`--layout auto|pip|split`、`--pip-x 0.655`、`--pip-y 0.338`（画中画左上角坐标/视频尺寸）、`--pip-vertical 7`、`--pip-horizontal 7`（越大越严格）、`--split-threshold 26`、`--merge-gap 1.5`（秒）、`--min-duration 0.8`（秒）。目前只支持顶部右侧小窗和中线附近左右分屏，其他布局请先用参数标定或扩展算法。

## 可选：本地视觉模型复核（实验性）

```bash
# 已安装 Ollama 时（没有则按 https://ollama.com/ 获取 macOS 版本）
ollama pull qwen3-vl:4b
# 确认 Ollama 本地服务已启动，模型可用
ollama list
video-incese --input "$HOME/Movies/source.mp4" --output "$HOME/Movies/incese-vlm" \
  --ollama-model qwen3-vl:4b --evidence
```

仅将候选片段的中间一帧提交至 `127.0.0.1:11434`。当 Ollama 不可用、模型缺失或输出无法解析时会明确报错，不会把模型结果伪装为已验证。Ollama 视觉模型是否正确判定动作对应关系仍需要实际人工标注数据验证。

## 输出契约

```text
incese-out/
  clips/               # 每个有效连续时间段输出一个可播放 MP4（无命中则不创建）
  evidence/            # --evidence 时每段一个中间帧（无命中则不创建）
  report.json          # 输入视频信息、参数、所有采样点分数、候选时间戳、导出结果
```

文件名包含原始 stem 和精确起止时间；默认覆盖相同文件名输出，建议不同参数使用独立输出目录以方便对比。原始视频不会写入。运行失败时检查 `stderr`；常见原因是 FFmpeg 缺失、编解码器不可用或无权写入输出目录。

## 验证要求与当前进度

- `pytest`：合成 PiP/分屏/纯单画面与时间区间合并测试。CI 在 macOS、Ubuntu 各两个 Python 版本上执行。
- 样本：四段上传样本用于开发回归，**都为正例**；不能用于证明真实负例精度。未经足够多真实正负例标注，不报告 Precision、Recall 或 F1。
- 人工验收：每条原始长视频人工标记起止点，比较脚本输出的起止点、误收和漏收；检查声音同步和 MP4 可播放。

## 已知限制

目前是基于 **画面几何结构** 的候选片段提取器。缺少视觉语义判断时，其他类型的画中画（游戏直播、人像主播、新闻播报）可能被保留；相同场景的低对比度小窗可能被漏检。`--ollama-model` 仅单帧语义核验，不保证完整片段每一帧都满足要求。对负例准确率和召回率应在用户本地真实长视频上评估后再考虑生产使用。