# RFC: 白模与渲染视频共现区间识别（研究与实施设计）

状态：**研究完成 / 待实现与真实数据验证**。这不是已交付的分类器，不改变当前默认运行逻辑。

## 目标和非目标

输入为包含白模、渲染视频、网页/软件教程、静态图、单独视频及混合画面的长视频。
只保留 **同一连续时间内白模动态视频与对应的渲染动态视频同时出现** 的原始 MP4 视频片段，保留原始分屏/画中画布局、声音和时序。

非目标：单纯镜头切分、只把截图导出为数据、只对每段抽一个中间帧判断。静态白模设定图与渲染视频的并排画面默认不合格。

## 已核验的相关研究及适用边界

1. [SEA-RAFT (ECCV 2024)](https://github.com/princeton-vl/SEA-RAFT)：
   稠密光流与跨帧运动估计。可辅助发现两个 ROI 中的连续运动、将背景/界面变化与主体运动区分；**不能仅凭光流证明白模与渲染画面的语义对应关系**。官方代码主要基于 CUDA/PyTorch，Mac CPU/MPS 适配必须独立验证；默认先用 OpenCV Farnebäck/稀疏特征法做基线。
2. [ActionFormer (ECCV 2022)](https://github.com/happyharrycn/actionformer_release)：
   时间动作定位与边界预测的架构参考；公开 THUMOS14 等 benchmark，并未提供本任务可直接部署的预训练分类器。
3. [TransNetV2](https://github.com/soCzech/TransNetV2)：
   镜头切换检测；只能辅助边界，不能将无转场的 PiP 消失作为镜头切换。
4. [SAM 2](https://github.com/facebookresearch/sam2)：
   跨帧目标掩码跟踪；作为布局漂移困难样本的可选升级模块，不作为 16GB Mac 默认依赖。
5. [OpenTAD (CVPR Workshops 2025)](https://github.com/sming256/OpenTAD)：
   时间检测架构和评估方法的参考，不直接安装其重型训练/推理依赖。
6. [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)：
   辅助镜头边界检测、时间编码和 FFmpeg 视频切分；不是目标内容分类器。

## 当前代码的问题

- `video_incese.py::classify` 仅从固定 PiP 起点/中线像素差识别布局。
- `video_content.py::_regions` 固定裁剪两个区域；一旦画面内容/窗口位置发生变化，误检或漏检都会发生。
- `_motion` 使用跨两帧灰度差>14 的像素占比，无法区分角色动画、滚动网页、光照/编辑器变化；不能证明“连续视频”。
- `verify_ollama_frames` 每次用 3 张抽帧提示 VLM，给出整窗 boolean；单窗判定不够细，无法保证时间连续性和前后边界。
- `semantic_intervals` 每次二分调用 VLM，容易在 8B 模型上超时/高成本，且 1 秒扫描可能漏掉短目标片段。
- `--evidence` 的 JPG 仅作为审查证据；正式输出必须在 `clips/*.mp4`。

## 建议的处理流水线（分层且可回放）

1. **分区布局**（CPU）：保留现有 PiP、split 作为 baseline；增加动态候选矩形定位。每帧记录两个 ROI 的像素坐标、布局类型、置信分数；未找到布局不代表视频语义为假，属于布局阶段漏检。
2. **两个 ROI 的独立时序检查**（CPU）：以 4–8 帧的时间窗口为单位，计算分区运动轨迹、光流/帧差稳定性、局部运动聚集度和场景连续性。动态文字、网页滚动、全屏转场、相机晃动要进入不确定状态，不能直接认定“视频运动”。
3. **内容语义**（Mac Ollama `qwen3-vl:*instruct`）：对有有效双运动证据的稀疏窗口进行语义复核，要求 3/更多帧跨时间验证白模、渲染内容和两者对应性。记录结构化五分类：`white_model_video`、`rendered_video`、`correspondence`、`software_ui`、`uncertain`，不要用 `match=true` 单字段覆盖错误细节。未知结果默认不导出正式数据，保留供人工复查。
4. **时间序列后处理**：保存每个窗口的多值状态及理由；禁止合并确定的负例窗口。只在有相邻时间证据的情况下填短缺口；小于窗口长度的目标片段不保证召回。
5. **边界细化**：粗扫使用 1–2 fps；仅在状态变化处使用局部更密集时序与场景检测，先基于 ROI/运动指标快速确定边界，再对极少数含混时间点调用 VLM。报告 `boundary_uncertainty_ms` 与采样步长，不能把二分收敛区间当成真实准确度。
6. **导出**：仅通过最终判定的 `[start,end)` 用 FFmpeg 重新编码 MP4；保持原始对比画面和可用音轨；输出 JSONL 逐窗口审计、report.json、可选 evidence/JPG，不因前置模型失败输出不合格片段。

## CLI 接口建议（尚未实现）

```bash
video-incese --input path/to/original.mp4 --output out/ \
  --content-mode temporal --layout auto \
  --ollama-model qwen3-vl:4b-instruct \
  --scan-fps 2 --temporal-window 2.0 --refine-fps 8 \
  --report-jsonl out/observations.jsonl
```

不要用 YAML 任务配置作为入口；独立提示词与 Schema 应版本化，并由 CLI 引用。

## 测试集与验收条件（不可省略）

- 用用户此前上传的**19 段导出视频**作为首批真负例/混合例回归候选，单独人工标注子区间；不能仅凭文件名当完整正负例。需拿到完整原片才能测试漏检召回。
- 正例：左右分屏白模/渲染双运动、PiP 双运动、低运动白模、较短有效片段。
- 负例：Blender 编辑、网页 UI、静态白模+渲染、纯白模、纯渲染、不相关双视频、游戏+摄像头、页面滚动、快速转场。
- 混合：同一候选中“有效→无效→有效”，要求输出两个 MP4，不包含已确定的无效窗口。
- 测量：逐时间段 Precision/Recall/F1、真实负例误收率、片段级 mAP@tIoU、边界起止 MAE、解码/导出成功率及 Mac 推理耗时。
- v0.2.1 不得声称完成以上指标；必须有带人工标签的真实原片+模型执行日志才能给出数值。
- 所有测试需保留输入哈希、模型标签、Prompt/Schema 版本、时间戳和运行参数，使失败可复现。

## 落地优先级

P0：写测试样本的真实标签，动态 ROI + 时间连续性基线，降低已确认静态/网页误收，严格的负例不桥接与 MP4 导出验收。

P1：稀疏 Ollama 语义模型审核、分层疑难处理和 VLM 请求缓存，避免在二分每一步都重复生成完整语义判断。

P2：如 P0/P1 无法解决跨布局移动、光流复杂场景，再评估 SEA-RAFT、SAM 2、ActionFormer/OpenTAD 作为可选增强。不得在 16GB Mac 的默认安装里强制引入这些训练/推理重量级框架。
