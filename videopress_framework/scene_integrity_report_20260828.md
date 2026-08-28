# 场景正确性与全流程审计（2026-08-28）

## 当前结论

场景边界修复和全量窗口审计已通过；正式的 18 类型官方全量模型评测将在同一口径下运行。最终评测只使用 guard 后的官方 `SceneLoader` 窗口，不使用旧的跨 scene 输出。

已确认：

- 0–5 号 GPU 用于并行评测，6–7 号 GPU 保持空闲。
- 官方筛选、官方 `VideoDriveFeatureBuilder`、Wan2.2 5B 推理、轨迹转换和 PDM 仍由官方 evaluator 负责。
- 新增代码只位于 `videopress_framework/`；没有修改 `third_party/navsim/`、`diffsynth/` 或官方 evaluator 源文件。
- 运行时拒绝没有 active scene token 的 pipeline 调用，防止复用上一场景 binding。

## 官方窗口与缓存覆盖

审计文件：`outputs/scene_window_audit.json`

| 项目 | 结果 |
|---|---:|
| `navtest.yaml` whitelist token | 12,146 |
| 未加 guard 的官方 lite loader 窗口 | 12,123 |
| 原始跨 `scene_token`/`scene_name` 窗口 | 4,247 |
| guard 后官方 `SceneLoader` 窗口 | 7,876 |
| guard 后窗口审计错误 | 0 |
| 完整 NVMe metric cache token | 12,146 |
| guard 后窗口缺失 cache | 0 |
| guard 后窗口集合与原始有效窗口集合一致 | true |

原始坏窗口的典型形式是 `frame_idx` 从某个 scene 的末尾跳到下一个 scene 的 0；例如 `[35, ..., 39, 0, ..., 9]`。guard 保留官方的 log、whitelist token、route 和 frame 参数，只过滤这类完整窗口，因此最终评测的正确场景数是 7,876，而不是 12,123。

## 筛选位置审计

官方 18 类型配置全部声明 `domain=last_history`。DriveVA layout 按 `video + trajectory` 构造，候选区由 layout 计算为：

```text
[history_video.end - tokens_per_latent, history_video.end)
```

每个 runtime event 会记录：`candidate_start`、`candidate_end`、`n_candidate`、`selected_global_min/max`、`selection_candidate_valid` 和 `selection_candidate_unique`。审计器会拒绝候选区外索引、重复索引、domain override、跨 scene 的 `scene_tokens`，以及不连续的采样帧。

已有框架/真实 hook 回归测试验证了：

- causal `last_history` 候选区为最后一个 history latent；
- physical KV prune 保留 protected token，并只从候选区选 K 个 token；
- K/V 输出维度和 mapping 与选中位置一致；
- probe score cache 的 scene token 与事件 token 一致。

## 已排除的旧输出

此前在未安装 scene-boundary guard 时产生的 `official_navsim_full_corrected`、`official_navsim_full_final` 和 `official_navsim_full_6gpu` 只作为故障定位证据保留，不纳入最终结果；它们可能包含 12,123 个原始 loader 窗口，不能证明没有跨 scene。

## 正式全量命令

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
/home/cpj/miniconda3/envs/DriveVA/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=6 \
  scripts/run_official_navsim_press.py \
  --metric-cache-path /mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_full \
  --score-cache-root /mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/official_press_score_cache_same_scene_6gpu_final \
  --output-root outputs/official_navsim_same_scene_6gpu_final \
  --force-full-scene-set --enable-nuscenes-metrics
```

正式目录应包含 18 个方法、每个方法 7,876 个官方 CSV token、事件和 records；完成后再运行：

```bash
/home/cpj/miniconda3/envs/DriveVA/bin/python scripts/check_scene_integrity.py \
  --suite-root outputs/official_navsim_same_scene_6gpu_final
```

最终必须满足 `ok=true`、`failed_method_count=0`、每个方法 CSV/event/records token 集一致、rank 分片一致、segment 审计无错误、`invalid_selection_count=0`，并且 suite 生成统计表与可视化目录。
