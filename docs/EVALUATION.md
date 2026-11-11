# video-incese 人工证据评估（v0.5.0）

## 标签契约

- `positives`：人工确认应当保留的**原始视频绝对时间**片段。
- `negatives`：人工已经查看并确认不该保留的时间片段。
- `UNKNOWN`：未标注的所有剩余时间。不能自动判为负例。

标注文件必须与原片对应；同类区间不可重叠，正负区间也不可重叠。不要从检测器自己的输出自动制造真值。

在本地创建文件（示例中的时间是假数据，并非原片2的真实边界）：

```json
{
  "source": "原片2.mp4",
  "positives": [{"start": 10.0, "end": 14.0}, {"start": 20.0, "end": 27.5}],
  "negatives": [{"start": 0.0, "end": 7.0}]
}
```

请将已人审的28段原片起止秒数填入 `positives`。如果某个空隙还没核实，只能保留为 UNKNOWN，不能放进 negatives。未有已核实负例时使用 `"negatives": []`。

## 对照命令

在源码目录激活 Python 3.12 虚拟环境、完成 `python -m pip install -e '.[dev]'` 后：

```bash
SRC="/Users/muse/Downloads/白模/原片2/原片2.mp4"
BASE="$HOME/Movies/video-incese-v050"
TRUTH="$HOME/Movies/video-incese-eval/truth-original2.json"

video-incese --input "$SRC" --output "$BASE/old-cuts" --content-mode temporal-cv \
  --boundary-mode scene --boundary-threshold .42 --boundary-min-segment .85

video-incese --input "$SRC" --output "$BASE/conservative" --content-mode temporal-cv \
  --boundary-mode scene --boundary-threshold .55 --boundary-min-segment 2 \
  --motion-diagnostics

video-incese --input "$SRC" --output "$BASE/no-cuts" --content-mode temporal-cv \
  --boundary-mode off

for name in old-cuts conservative no-cuts; do
  video-incese-eval --report "$BASE/$name/report.json" --truth "$TRUTH" \
    --output "$BASE/$name/metrics.json" >/dev/null || exit 1
done
```

如果此前已经有 `report.json`，无需重新编码，直接运行评估 CLI 即可。

## 如何理解报告字段

- `positive_time_recall`：被预测片段时间并集覆盖的人工正例时长 / 人工正例总时长。
- `one_to_one_iou_matches`：与人工案例的 IoU≥阈值（默认0.5）的**最大数量的一对一匹配**。一个长预测视频不能同时充当多个完整人工案例。
- `matched_boundary_mae_seconds`：仅计算成功 IoU 匹配片段的起止边界平均绝对误差，不是全局边界误差。
- `split_positive_cases`：同一条人工正例与多条预测片段至少各重叠0.25秒的数量（过切线索）。
- `merged_positive_cases_in_prediction`：一个预测片段与至少两条人工案例各重叠0.25秒的数量（错合并线索）。
- `positive_clips_95pct_covered`：正例时间覆盖率达95%的人工案例数，不代表切段边界正确。
- `confirmed_negative_predicted_seconds` 和 `confirmed_negative_false_accept_rate`：已经人审的负例中被预测命中的时长与比例。
- `precision_on_labeled_time_only`：仅已标注正负时间区域的命中精度；没有负例或没有命中时为 null，不能外推整个视频。
- `global_precision: null`：未标注时间的真实类别未知，所以不能声称完整精度已知。

评估工具将 `exported` 和 `dry_run` 作为预测状态，排除 `export_error` 和 `rejected_*` 等失败或拒绝状态；分析实际导出数量时要单独看导出失败。

## 风险与验收

本次修改遵循报告建议，优先工程评估和防过切，不引入未经验证的硬语义拒绝。可选的 `motion_correspondence` 只是双区域运动幅度相关系数，相关性高也可能只是镜头同时变化；**不证明白模动作与渲染动作匹配**。

原片2（28条人工剪辑）已参与开发与调参，只适合开发回归，不能用来证明泛化效果。比较时同时看正例召回、已知负例误收、IoU匹配、过切和错合并、MP4数与运行时间，不能只看是否切得更少。保留至少一条从未用于训练或调参的新原片，按**原始视频**而不是切片划分数据集。

研究报告提出未来可探索 OpenCV ORB、光流轨迹、PySceneDetect、轻量嵌入；这些只能在人工标注数据上实测后决定是否值得引入。不应为了“匹配度低”而未经验证就自动删除正例。

资料：
- [PySceneDetect](https://www.scenedetect.com/docs/head/api.html)
- [OpenCV 特征匹配](https://docs.opencv.org/4.13.0/dc/dc3/tutorial_py_matcher.html)
