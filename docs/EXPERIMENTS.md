# v0.7 并行管线实验（opt-in，不修改默认生产算法）

## 为什么做这轮实验

已知 v0.6 的 `原片2` 正例特征库包含 28 个人工案例 / 357 个特征样本，并在同一源的排序自检中达到 Top50 正例时间召回100%，但只与 19/28 个案例实现一对一 IoU≥0.5。训练源重检不是盲测，也未证明未标注片段不合格。

目标：将 **相似度排名** 和 **片段组织** 分开，在相同窗口预算下重复检验几个方案。避免因某个方案导出更多视频就假称效果更好。

## 管线

1. **balanced**：v0.6 基线；在每个人工案例内找单个最近样本，跨前三个相似案例聚合。
2. **nearest**：完全依赖最相近的单个案例/样本，容易被孤立噪声影响。
3. **top3**：在每个人工案例中取三个最近样本距离的均值，再在各案例之间聚合；抑制单样本偶然吻合。
4. **centroid**：每个人工案例的每个布局取特征中位数构成中心点；减少局部噪声，但可能遗漏多阶段动作。
5. **motion**：只比较运动、光流、运动一致性、双区运动比例和切镜头比例，不使用颜色与边缘密度；用于检验“运动证据是否足够”。

每条路线还测试 `off` 与 `median3` 两种平滑方式。median3 只在连续的三个扫描窗口上做中位数平滑，不跨时间轴空缺。实验不改变人工正例，不增加伪造的负例标签。

## 快速离线复跑：无需原视频、不需要再次耗费约4分钟解码

```bash
cd /Users/muse/code/video-incese
git pull --ff-only origin main
source .venv312/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q

OUT="$HOME/Movies/video-incese-v060"
TRUTH="/Users/muse/Downloads/白模/原片2/原片2.json"

video-incese-experiment \
  --ranking "$OUT/original2-scan/ranking.json" \
  --profile "$OUT/positive-profile.json" \
  --truth "$TRUTH" \
  --output "$OUT/offline-experiments" \
  --top-k 10 20 50
```

**本次实测发现 v0.6 相似度阈值0.62接受了477/478个窗口（99.8%），导致几乎整片合并为2个超长片段，时间召回100%但一对一案例IoU匹配为0。那一组结果不能据此比较排序策略。**

实验工具现已修正：**默认自动比较每个方法前10%、20%、30%、40%的窗口**，而不是拿几乎整片通过的0.62阈值反推窗口预算。对于478个窗口，对应约48/96/143/191个窗口；每档都在5种策略和两种平滑方法之间使用相同预算。输出增加 `selected_window_positive_recall`（筛选窗口层的正例召回）与 `positive_time_recall`（合并成候选片段后、Top-K层的正例召回），避免把合并成长片误认为检索精准。

可以通过下面的命令显式控制不同的窗口预算，推荐比较50/100/150/200，而**不是477**：

```bash
video-incese-experiment \
  --ranking "$OUT/original2-scan/ranking.json" \
  --profile "$OUT/positive-profile.json" \
  --truth "$TRUTH" \
  --output "$OUT/budget-sweep" \
  --budgets 50 100 150 200 \
  --top-k 10 20 50
```

如果只要单个预算，也仍支持旧选项 `--budget-windows 80`。

如需控制相同的候选窗口预算，例如只允许 80 个窗口：
```bash
video-incese-experiment \
  --ranking "$OUT/original2-scan/ranking.json" \
  --profile "$OUT/positive-profile.json" \
  --truth "$TRUTH" \
  --output "$OUT/window80-experiments" \
  --budget-windows 80 \
  --top-k 10 20 50
```

如果 `ranking.json` 是用 v0.6 生成，它只保存了原先选中的 PiP 或 split 布局特征，因此**重算各策略时使用固定区域假设**；新版本扫描才会同时保存两个布局，供公平重放。

输出：
- `experiments.csv`：按 5 种策略×2 种平滑方式×多个窗口预算×TopK 输出的对照表。
- `experiments.json`：样本预算、是否有完整布局假设、各策略 Top10 候选时间、召回/一对一 IoU/拆分/合并指标，以及验证限制。

**公平约束**：所有方法选择的窗口数量相同；不同的相邻窗口如何合并仍会造成最终 MP4 数量/总时长不同。实验统一关闭视觉场景切割，以便先单独比较检索排序。实际“案例从哪里开始和结束”属于下一阶段边界专门实验。若需要正式输出，另用 `video-incese-rank scan` 进行实际视频扫描。

## 运行另一条完整视频扫描管线

```bash
SRC="/Users/muse/Downloads/白模/原片2/原片2.mp4"
video-incese-rank scan \
  --input "$SRC" \
  --profile "$OUT/positive-profile.json" \
  --output "$OUT/top3-full-scan" \
  --strategy top3 \
  --min-similarity 0.62 \
  --boundary-mode scene \
  --export-top 20
```

**重要**：各方法的分数分布不同，所以固定阈值0.62不能直接构成公平优劣对照。先通过离线同窗口预算实验确定值得测试的方案，再在新原片盲测并单独校准阈值。原片2只是开发数据。

## 标注冲突审计（可选）

之前的人工复核 CSV 中，部分 `reject` 时间与最新 28 个人工剪辑时间有重叠。如果这些文件的原始视频与时间轴一致，存在标签冲突，需要人工明确。不能自动把先前的 reject 改成 keep，也不能因 28 段外没有剪出就判为 reject。

```bash
video-incese-experiment \
  --ranking "$OUT/original2-scan/ranking.json" \
  --profile "$OUT/positive-profile.json" \
  --truth "$TRUTH" \
  --review-labels "/path/to/review_labeled_complete.csv" \
  --output "$OUT/label-audit"
```

输出的 `label_audit.possible_time_axis_conflicts` 会列出候选冲突，`source_identity_verified=false` 表示程序无法通过 review.csv 本身核实原视频 SHA256；应先确认是否为同一原始时间轴。

## 场景边界的另一条独立实验

对于已有的 v0.4 或 v0.5 report.json，以及人工原片2.json，可以离线比较切点阈值：

```bash
video-incese-boundary-bench \
  --report "$HOME/Movies/video-incese-v040/report.json" \
  --truth "$TRUTH" \
  --output "$OUT/boundary-sweep.json" \
  --thresholds 0.42 0.50 0.52 0.55 0.60
```

请把 `--report` 替换成你机器上实际存在的 v0.4.0 报告路径。**报告中必须包含候选场景事件及其得分**；使用 v0.5 报告时，已在0.55阈值下根本没有提出的低分切点不能从 JSON 中重新恢复。

这个实验只把**两个相邻人工案例首尾直接相接**的时间定义为已确认的“案例间切点”。其余人工案例的边缘可能连接 UNKNOWN，不应该当作肯定的切点。还会将被接受的视频中未覆盖到的真值边界标为不可观测，不当作场景检测器的漏检。

输出：不同阈值的 `matched_true_boundaries`（对人工案例相接处的有效命中）、`interior_split_proposals`（人工完整案例内部切点）及详细秒数；只是已有切点的**离线筛选**，不是重新运行算法，不能据此自动更新默认参数。

## 验收边界

**CI 能证明**：流程对有效输入有可执行输出、同预算控制、历史输入兼容性与防止未标注片段误当负例。**真实效果必须由原片与人工标注验证**：

- 相同窗口预算下比较 `annotated_positive_time_recall`、`one_to_one_iou_matches`、过切/合并数量；不可只比较某个最高分或 TopN 数量。
- 按不同原视频分组盲测，训练原片2对自己的检索不算独立测试。
- 人工复核新视频的 Top20/50 及低排名随机样本，才可统计 Precision@K。
- 不把 `similarity` 当作合格概率；也不把视觉切点当成真实案例边界。
- 如果没有可证明的独立提升，继续保留默认 v0.6 `balanced` 与旧版 `video-incese` 管线。
