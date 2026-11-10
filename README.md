# video-incese

自动提取 **参考动画/三维模型与对应生成视频同时出现** 的长视频区间，支持画中画和左右分屏。目标是内容共现而非单纯镜头分割。macOS（Apple Silicon/Intel）与 Ubuntu 均可运行。原文件不改动。

## v0.2 核心变化

旧版只识别画面几何布局，网站操作、Blender、静态角色设定图都会误收。**v0.2 默认启用语义模式**，通过三层过滤：

1. **布局初筛**：OpenCV 检测右上 PiP / 居中左右分屏（保留原来两个阈值检测器）。
2. **双区域运动 + 多帧内容识别**：同一区间抽取前、中、后 3 帧，分别衡量两个画面区域的运动；两个区域都具有时间变化才交由本地 Ollama 视觉模型，模型按角色/动作对应、两边是否视频、是否软件教程等进行判定。仅凭分屏线不会通过。
3. **边界细化**：默认每 1 秒做语义判断；有正负判定切换时，对时间点二分重检，默认缩小到 0.125 秒的判定区间；**遇到明确拒绝窗口绝不跨越合并**。这是基于有限采样的边界估计，不是逐帧人工真值，也不能证明每一帧都满足条件。

## macOS 安装

```bash
brew install python@3.12 ffmpeg git
git clone https://github.com/Jqsfree/video-incese.git
cd video-incese
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

语义模式依赖本机运行 Ollama，模型由用户明确下载；**无需 CUDA、无云端密钥**。安装 Ollama 后：

```bash
ollama pull qwen3-vl:4b
ollama list
```

## 先只检测，再实际导出

```bash
# 推荐：严格内容模式。默认 content-mode=semantic，缺失模型会明确报错而不是退回几何模式
video-incese --input "$HOME/Movies/source.mp4" \
  --output "$HOME/Movies/incese-check" \
  --ollama-model qwen3-vl:4b --dry-run --evidence

# 查看 report.json 中的 clips 和 verification_trace，确认后实际导出
video-incese --input "$HOME/Movies/source.mp4" \
  --output "$HOME/Movies/incese-output" \
  --ollama-model qwen3-vl:4b --evidence

# 需要更多时序采样（更慢、更贵）
video-incese --input "$HOME/Movies/source.mp4" \
  --output "$HOME/Movies/incese-dense" \
  --sample-fps 4 --semantic-step .5 --boundary-tolerance .125 \
  --ollama-model qwen3-vl:4b --evidence

# 旧算法只作基线调试；不应当视为有效内容分类
video-incese --input "$HOME/Movies/source.mp4" \
  --output "$HOME/Movies/incese-layout-only" --content-mode layout --dry-run
```

输入路径支持中文与空格。每次输出到独立目录以避免旧文件混淆。若已克隆仓库，使用 `git pull --ff-only origin main` 更新，并重新执行 `pip install -e '.[dev]'`。

## 输出契约

```text
incese-output/
  clips/             # 通过语义检查的连续片段 MP4；无命中时不生成
  evidence/          # 抽帧证据；被拒绝候选 *_rejected.jpg
  report.json        # 几何采样、语义探测、起止时间、状态、错误
```

`report.json` 中的 `verification_trace` 记录每个源候选的多帧判定：`time`、`match`、`layout`、`reference_motion`、`generated_motion`、`reason`、`phase`（scan/boundary）。`clips[].source_candidate` 可追溯原始布局区间；一段混合候选可以分成多个 MP4。

默认参数：`--sample-fps 2`（布局）、`--semantic-step 1`（语义时间步长）、`--motion-delta .3`（三帧时间半径）、`--motion-min .02`（运动像素比例）、`--boundary-tolerance .125`、`--min-duration .8`、`--merge-gap 2.6`（仅影响原布局候选预筛，语义不跨拒绝窗口合并）、`--layout auto|pip|split`、`--pip-x .655`、`--pip-y .338`。

## 故障与质量边界

- `frame_error` / `verification_error` / `verification_skipped` / `export_error`：记录在 report，退出码 **1**。模型失效不会自动转成布局模式或被当作匹配。
- 视频末尾未分析超过允许范围：`sampling.incomplete=true`，退出码 **2**；少量孤立解码错误会计入警告。
- `rejected_semantic` 不是程序故障；`--dry-run` 不导出视频。
- 几何候选不会覆盖所有布局，尤其是不同位置的画中画、无明显边框或其他分屏形式。
- 运动门槛会牺牲极低运动参考片段的召回率；摄像机移动、转场也可能造成虚假的运动证据，故仍须视觉模型判断。
- 视觉模型输出不是人工真值，甚至多帧仍可能误判教程、静态参考或动作对应；**本地实际模型效果和真实端到端 Precision/Recall 尚未经过独立验证，不能声称生产验收通过**。
- 当正负内容在一个语义采样间隔内快速交替，可能漏掉过短片段；建议缩短 `--semantic-step` 并检查 `verification_trace`。对高价值片段建议人工验收，最大时间精度不能直接等于边界误差。
- 输出采用 FFmpeg 重新编码，保留可用音轨并使用精确时间戳；输入视频不写入。

## 验收

先对包含真正正负例的完整长视频标注区间，至少区分：真实模型对照、工具网页、Blender 编辑器、静态参考图、单独生成片段、不相关双视频和正负混合片段。检验误收、漏收与边界偏差，并保留 `report.json` 与 `evidence/`。合成测试和组件级测试并不能证明实际 VLM 精度。
