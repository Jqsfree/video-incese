# v0.6：基于人工剪辑特征的相似度检索和优先排序

## 一、流程和使用边界

```text
人工已核实的 28 段原片绝对时间（原片2.json）
    ↓ 读取原始视频，按每 2 秒/每秒 4 帧取样
每个人工案例的多窗口正例特征库：11 项时序/光流/边缘特征 + 3 项区域外观特征
    ↓
新长视频全时间轴扫描（不依赖旧版 PiP/分屏候选，layout=none 也计算）
    ↓
每个窗口对照人工正例，算与最接近案例的相似度
    ↓
满足候选阈值的相邻窗口合并，场景切点可选进行细分
    ↓
按相似度从高到低排序，人工先审高排名，必要时导出最高排名 MP4
```

这是**正例检索与排序**，不是已经具备可靠语义理解的自动剪辑分类器。两区同时运动、颜色接近、光流相似等都可能来自无关双视频；视觉场景切点也可能位于同一动作内部。相似度 0.9 **不等于 90% 合格概率**。未标注时间严格属于 UNKNOWN，不是负例。保留旧版 `video-incese` 基线（不变更其默认行为）。没有使用或默认启用 model-v1。

## 二、在 Mac 上建立 28 段正例特征

在已完成 `source .venv312/bin/activate` 与 `python -m pip install -e '.[dev]'` 的项目目录：

```bash
SRC="/Users/muse/Downloads/白模/原片2/原片2.mp4"
TRUTH="/Users/muse/Downloads/白模/原片2/原片2.json"
OUT="$HOME/Movies/video-incese-v060"

video-incese-rank build \
  --source "$SRC" \
  --truth "$TRUTH" \
  --output "$OUT/positive-profile.json"
```

`TRUTH` 必须指向你已有的“原片2.json”，里面的 `clips[].start_ms/end_ms` 代表**原片绝对时间**；也支持 v0.5 的 `truth-original2.json`（`positives[].start/end`，秒）。如文件不在示例位置，应替换本机真实路径，程序不会搜索或猜测。

报告将显示 `reviewed_cases`（期望 28）、`prototypes`（各正例的局部特征数量）、`geometry_fallback_windows`（人工正例虽成立但旧几何检验失败，强制尝试双区域模板的窗口数）、`decode_failures`。如果有人工正例完全解码失败，构建会失败，**不允许把未提取特征的案例悄悄当成学习成功**。输入来源文件 SHA256 和人工标注文件 SHA256 保存于配置，便于核验。

注意：仅有 report.json 无法对 28 段提取完整特征，尤其旧版布局过滤掉的案例 12/15/18；`build` **必须读取真实原始 MP4**。

## 三、扫描未处理视频、进行相似度排序

```bash
NEW_VIDEO="/path/to/new-unprocessed-video.mp4"

video-incese-rank scan \
  --input "$NEW_VIDEO" \
  --profile "$HOME/Movies/video-incese-v060/positive-profile.json" \
  --output "$HOME/Movies/video-incese-v060/new-scan" \
  --min-similarity 0.62 \
  --boundary-mode scene
```

输出：

- `ranking.csv`：按相似度**降序**，带名次、原片起止时间、长度、最相近的人工案例 ID，以及人工复核状态。
- `ranking.json`：全部窗口的特征、分数、匹配案例证据、布局分类、原片扫描失败记录和场景切点。
- 默认**不导出视频**，防止对 955 秒以上的长视频做多余编码。明确要看实物时，在扫描命令中加 `--export-top 20`，只生成排名前 20 的 MP4。

这里的 `--min-similarity .62` 只是候选发现的**起点参数**，不是经人工验证的质量阈值。低于该值的窗口也仍记在 `ranking.json.windows` 中。想扩大候选召回，尝试 `.50` 并比较新增人工片段覆盖；不能直接宣称提升。

如果先用原片2自检，再扫描新视频：

```bash
video-incese-rank scan \
  --input "$SRC" \
  --profile "$OUT/positive-profile.json" \
  --output "$OUT/original2-unreviewed" \
  --exclude-reviewed \
  --boundary-mode off
```

`--exclude-reviewed` 只在**原片字节 SHA256 与训练来源一致**时有效，并跳过与已确认正例重叠的窗口；其余时间仍为 UNKNOWN。该模式用于“找原片2里人工 28 段之外的可能漏选内容”，**不能把检索结果评为独立泛化准确率**。

## 四、为何不是普通镜头切割

`build` 从人工原片采样获得以下特征：

- 运动：参考区和渲染区各自的灰度帧差、光流、运动方向一致度、同步运动比例
- 外观：参考区边缘密度、低饱和度/亮度比例，以及渲染区低饱和度比例
- 时序：场景突变比例

相似度使用**每个人工案例分别取最近的特征样本**，再组合最相近的几个案例，而不是把 46 秒长案例当成比 5 秒案例具有更多标签的“训练样本”。各特征使用人工案例统计的稳健尺度；权重只是工程先验，未经过独立测试学习。

如果旧算法 `classify` 返回 `none`，新管线会对固定 PiP 与左右分屏区域分别提取特征、计算相似度，选出更相近者。这样不会在**布局入口**直接漏掉整段，但固定区域不一定对应真实画面；新的检索结果必须人工复核。

候选时间的组织：相邻高分窗口合并；可选 `--boundary-mode scene` 再使用原有的双区域视觉突变切点拆开。**未实现经人审证明可靠的“同一动作案例连续性”识别。** 因为“两个镜头显著改变”不等于“案例改变”，不能仅按相似度排序就宣称完整案例边界已经解决。可以先用 `--boundary-mode off` 与 `scene` 对照。

## 五、验证方式与下一步

```bash
python -m pytest -q
video-incese-rank --help
video-incese-rank build --help
video-incese-rank scan --help
```

1. 检查构建日志：28 段是否全部提取成功，特别是之前没有进入时序检测的案例 12、15、18。若 `geometry_fallback_windows` 大于零，说明确实执行了布局兜底，但不能以此证明选对了区域。
2. `scan` 的 `geometry_none_but_scored` 说明有多少旧版未处理的窗口被纳入新排序；人工确认前不能认定新增窗口有效。
3. 将 rank Top 20/50 与随机低排名窗口分层人工复核，记录每段 `keep/reject/unsure`；别把未看过的低排名当 reject。
4. 本版本新增 `video-incese-rank evaluate` 专门评估相似度排名结果（见下文）。`video-incese-eval --report ...` 仍只评估旧版 report.json，不能把 ranking.json 当旧 report。Top-K 的人工正例时间召回率可以计算，但 **Precision@K 在未标注预测仍是 UNKNOWN 时严格保持 null**，必须人工检查高排名候选才可能计算真实精度。
5. 在**完全不同的原始视频**上做盲测；原片2用来构建正例时，原片2上的相似度结果是自检，不是可作为测试集的成绩。

如果排序对人工样本确实有效，再做下一阶段：利用被确认的正/负例校准阈值、优化真实区域位置、学习同一案例内部镜头变化与案例间边界。不要直接让未验证的高相似度自动生产正式交付数据。


## 六、Top-K 召回评估（不需要重新扫描视频）

在原片2完整扫描的结果上评估不同审查预算下覆盖了多少已确认案例：

```bash
video-incese-rank scan \
  --input "$SRC" --profile "$OUT/positive-profile.json" \
  --output "$OUT/original2-all" --boundary-mode scene

video-incese-rank evaluate \
  --ranking "$OUT/original2-all/ranking.json" \
  --truth "$TRUTH" \
  --top-k 10 20 50 100 \
  --output "$OUT/original2-all/evaluation.json"
```

注意：**不要**在做原片2已知 28 段的正例召回对照时加 `--exclude-reviewed`，否则已知正例已经被排除，召回天然接近零；`--exclude-reviewed` 只用于寻找原片2未审查区域的新候选。因原片2就是特征库来源，这仍是训练源上的开发自检；不能称为独立准确率。

输出包含 `annotated_positive_time_recall`、`one_to_one_iou_matches`、`split_positive_cases`、`merged_positive_cases_in_prediction`，以及 `precision_at_k: null`。如需质量排序指标，必须把高排名和低排名都抽样人工审查，不能把未审过的结果视为不合格。
