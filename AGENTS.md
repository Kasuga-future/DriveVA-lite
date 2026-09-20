# AGENTS.md — DriveVA-lite Video Token Compression 交接文件

> 最后更新：2026-09-20 23:50 CST
> 当前分支：`main`，当前 HEAD：`d6a423d`（已与 `origin/main` 同步）
> 当前工作区：**clean**，唯一 untracked 是排除项
> `videopress_framework/scripts/pre_dit_gpu_smoke.py`（见 §6）
> 当前主任务：Future token compression 现状：
> 1. future 直接 select：Phase F0 已打通；噪声上选 future 与随机接近，hard prune 明显掉 PDM。
> 2. **history-guided future select**：已实现 `HistoryGuidedFutureSelector`、
>    `all_video_history_only` scorer、`future_keep_ratio` cap 和 runner CLI；
>    做过 64 / 1024 / common-512 paired POC。
>    结论：`same_latent` 压缩近似乘 2 但 ΔPDM `−0.0280`；`union_history [0.05,0.40]`
>    仍 ΔPDM `−0.0121`；threshold/future-keep-ratio frontier sweep 后，唯一近无损点
>    `[0.02,0.40]` 压缩更少且更慢，没有找到 near-lossless 的乘 2 工作点。
> 3. **oracle future subset 上界（已完成）**：random-mask 上界 + multi-seed
>    trajectory/PDM tile oracle 均已完成，**tile oracle 失败**（512 面板 keep0.5
>    ΔPDM `−0.0424`；4-seed `combined keep0.5` ΔPDM `−0.0182`）。结论指向 **tile
>    粒度本身**：tile 子集空间可能不包含 near-lossless token 子集。
> 4. **token-level set-level oracle（已完成，代码 + 单测 + 1024 场景实验）**：
>    - `videopress/oracle/`：token 分组（token/linear/block/random）+ set-level
>      联合打分搜索（best-of-N random / greedy forward / greedy backward / beam）
>      + per-scene best-of-N 聚合 + trajectory displacement / planning harm；
>    - `oracle_future_token_mask` selector（显式 token 索引 mask，物理 kv_prune）；
>    - runner `--future-oracle-token-mask-json` / `--future-oracle-token-mask-jsons`
>      （一个候选 mask 一个 method，整批候选一次 runner 调用）；
>    - `scripts/search_future_token_set_oracle.py` 搜索驱动。
>    **结论：future hard prune 不可部署（1024 场景 keep0.5 随机 `−0.0483`）**，见 §4。
> 5. **history+future 联合 press（已完成 7/7 臂 + matched-K 随机对照）**：
>    `union_history` K=1149 / `same_latent` K=989 等联合臂前沿单调无拐点；
>    **联合 selector 相对 matched-K 全 video 随机剪枝有显著正贡献**
>    （`union_history` paired `+0.023` CI 不含 0），但绝对水平仍只有 `−0.016`，
>    **联合最佳可部署点仍然是 history-only**（`−0.0039`），见 §4 / §12。
> **资源约束（用户 2026-09-20 23:15/23:2x 明确要求）**：最多同时占用 **2 张 GPU**，
> 且必须**只使用真正空闲的卡**（free ≥ 40 GiB）；没有空闲就等待。
> **提交约束**：阶段性成果需用户发话后再 commit。
>
> **维护要求（强制）**：以后每个 agent 会话结束前，必须更新本文件：
> 1. 更新顶部“最后更新 / HEAD / 工作区”；
> 2. 在 §12 追加一条会话交接记录；
> 3. 新结果写入 §4；能力矩阵变化同步 §5；推翻旧结论时保留旧结论并标 `[已被新结果推翻]`；
> 4. 代码/实验状态变化后同步 §6–§8；
> 5. 不要把未验证的 hypothesis 写成 verified；不要删除失败实验，失败信息同样重要；
> 6. 如果本文件与代码/测试冲突，以代码和可复现测试为准，并立即修正本文件。

---

## 0. 给下一个 agent 的 30 秒摘要

- 项目根目录：`/mnt/chenpeijian/autodrive/DriveVA-lite`。
- 主框架：`videopress_framework/`，采用 `Domain + Scorer + Selector + Operator + Budget +
  Injection Point + Persistence` 的插件化结构；设计文档见 `press.md`，当前操作说明见
  `videopress_framework/README.md`。
- DriveVA 底层已在 `diffsynth/pipelines/wan_video_new.py` 中加了 runtime-only hook：
  `VIDEO_INPUT`、`BLOCK_INPUT`（pre-DiT hidden prune）、`SELF_ATTN_KV`（post-RoPE K/V prune
  + cross-layer persistence）。
- 当前最佳 history Press：`learned_planning_selector + history_threshold[0.05,0.40] +
  layer 15 + hidden_sequence`，全量 7,876 场景 PDM `0.911143`，平均保留 `490.964/780`
  （62.94%），延迟约 `568.38 ms`（NoPress PDM `0.909839`，约 `589–592 ms`）。
  绝对提升 `+0.001304`，95% CI 跨 0；结论是 **近无损加速，不是能力提升**。
- History token 有可利用冗余，但 Attention/Gradient 与 planning attribution 的一致性弱；
  时间连续性、跨层连续性都只略高于随机，因此不要假设同一批 token 长期有效。
- `future_video` / `future_latent_i` 已进入官方 runner 与框架 selector；learned selector
  已支持 future storage / history-compatible 两种位置模式。官方 single-scene smoke 已通过。
- **Future 64-scene POC（2026-09-17，truncated，不能当全量结论）**：在 `future_video`
  Layer 15 + `hidden_sequence` 下，`action_attention_vnorm` topk keep 50% 的 ΔPDM 约
  `−0.039`，keep 68.6% 约 `−0.027`；history-trained learned selector 零样本在 K≈535
  时约 `−0.065`（history_compatible）和 `−0.086`（storage），同预算下未优于 matched
  random。position mode 有影响（history_compatible 约 +0.021 PDM），但不足以抵消掉点。
  详见 `videopress_framework/outputs/future_poc64_report_20260917.md`。
- 未来 token 很可能是 planning 的关键输入：报告里 DriveVA `Video+Action=90.9 PDMS`，
  `Action Only=47.0`；上述 POC 与这个警告一致。下一步应先做 future oracle/上界分析，
  再决定是否投入 future selector 训练，不能直接上 hard prune。
- **资源约束（2026-09-20 23:15 更新）**：最多同时占用 **2 张 GPU**，且只使用真正空闲的卡
  （free ≥ 40 GiB，优先 0/1）；没有空闲 GPU 就等待，不与他人共卡。本文件 §3.1/§10 已同步。

---

## 1. 研究目标与当前结论

### 1.1 DriveVA 与 Press 的定位

DriveVA 在共享生成过程中联合 decode future video latent 和 action/trajectory token。
本项目不是重新设计世界模型，而是在推理时删除/压缩 video token，目标是：

1. 在不训练或轻量训练的前提下降低 DiT 计算量；
2. 尽量保持 trajectory planning 的 PDM；
3. 明确区分：
   - **causal / perturbation**：只替换 token 内容，不缩短长度；
   - **physical compression**：真正缩短 sequence length，减少 K/V、residual、RoPE、t_mod 和 MLP 计算。
4. 历史结论：Press 只适合利用模型已形成的表示冗余做近无损加速，不承担提升能力的任务；
   “能力提升”必须与 Press 解耦。

### 1.2 关键结论（已验证，按重要性）

- **冗余存在**：Layer 15 双 latent Press 平均保留约 490.96/780，严格面板加速
  2.84%–3.41%；history token 存在可删/可压缩的表示冗余。
- **不能稳定追踪关键 token**：相邻 latent 排序一致性只有约 0.5318；Layer 15→25 的
  Top-K overlap 约 0.5061，已接近随机。token 是固定 spatial patch，不是有身份的对象。
- **删除 token 没有证据能提升规划能力**：所有全量正向 ΔPDM 的置信区间都跨 0；
  greedy merge 在全量上显著下降；RegisterBottleneck 严重失败。
- **Pre-DiT 压缩 history 不可行**：在语义形成之前做 selector，读取的仍是贴近 VAE patch
  的特征，提前判断 30 层后的规划作用，效果差。
- **当前最佳部署方向**：Layer 15 选一次，Layer 16–29 跑短 residual，Head 前恢复布局；
  history threshold/attention-vnorm 做近无损加速。
- **Attention-VNorm 是当前最稳的免训练 scorer**：重要性 `score_i = alpha_i * ||V_i||_2`，
  动作 query 对历史 token 的平均 attention 乘以对应 value norm；但“分数高低”不等于
  “planning attribution 高低”。
- **未来 token 的特殊风险**：future video token 是 planning 的关键输入，报告中的
  Action Only 对照（47.0 vs 90.9 PDMS）说明直接删除 future video token 很可能大幅掉点。
  必须重训或重新设计结构化 attention，且要有 oracle 上界验证。

---

## 2. 代码地图

```text
DriveVA-lite/
├── AGENTS.md                         # 本文件，agent 交接与任务状态
├── README.md                         # DriveVA 官方 README
├── press.md                          # VideoTokenPress 设计文档（伪代码级）
├── diffsynth/
│   ├── pipelines/wan_video_new.py    # 核心训练/推理 model_fn；旧改动集中处
│   ├── trainers/utils.py             # 训练日志/指标透传
│   └── models/                       # Wan DiT、VAE、trajectory head 等
├── examples/wanvideo/
│   ├── driveva_train/                # NAVSIM 训练、online selector teacher
│   │   ├── train_navsim_v1.py
│   │   └── scripts/train_navsim_v1.sh
│   └── driveva_infer/                # NAVSIM/nuScenes/Bench2Drive 推理评测
│       └── eval_navsim_pdm.py
├── videopress_framework/
│   ├── README.md                     # 当前框架操作说明，优先阅读
│   ├── videopress/
│   │   ├── core/                     # layout/domain/budget/runtime/persistence/retention
│   │   ├── scorers/                  # random/norm/attention/gradient/learned selector
│   │   ├── selectors/                # topk/threshold/history/adaptive
│   │   ├── operators/                # zero/replace/kv_prune/hidden_prune/merge
│   │   ├── presses/                  # scorer_press/merge/register_bottleneck
│   │   ├── adapters/driveva.py       # 对 DriveVA pipeline 的 runtime hook
│   │   └── training/online_selector.py # 训练侧 selector、teacher loss、probe 工具
│   ├── evaluation/                   # evaluator/statistics/visualization
│   ├── configs/press/                # 预置 press 配置
│   ├── scripts/                      # 官方 runner、full suite、分析脚本
│   ├── tests/                        # 184 个测试当前全部通过（2026-09-17）
│   └── outputs/                      # 大量实验与报告
└── outputs/                          # 早期 smoke / round 产物
```

### 2.1 最相关的核心文件

| 文件 | 作用 |
|---|---|
| `videopress_framework/videopress/core/layout.py` | DriveVA token 布局；`history_video`、`future_video`、traj/future_action 范围；`build_driveva_layout` |
| `videopress_framework/videopress/core/domain.py` | `last_history` / `history` / `all_video` / **`future_video`**；当前 future latent 单独 domain 尚未定义 |
| `videopress_framework/videopress/core/budget.py` | 绝对/比例 budget；reference `eligible/history/last_history/video/future_video` |
| `videopress_framework/videopress/core/runtime.py` | Hook 生命周期、probe/score cache、事件记录、selection audit |
| `videopress_framework/videopress/core/persistence.py` | source layer 选一次，后续层复用 selection；`hidden_sequence` 会同时缩短 residual/RoPE/t_mod |
| `videopress_framework/videopress/adapters/driveva.py` | 把 press 挂到 `pipe.dit`/`pipe.dit2`；`BLOCK_INPUT` 与 `SELF_ATTN_KV` 都在这 |
| `videopress_framework/videopress/scorers/attention.py` | `action_attention` / `action_attention_vnorm`；用 future_action 做 query，可天然作用于 future domain |
| `videopress_framework/videopress/scorers/learned_selector.py` | 部署侧 learned selector；当前 `_positions()` 明确要求 history domain |
| `videopress_framework/videopress/training/online_selector.py` | `DynamicTokenSelector`、gradient/ signed / displacement teacher、训练 metrics |
| `examples/wanvideo/driveva_train/train_navsim_v1.py` | online selector 训练循环；counterfactual probe、history mask、日志指标 |
| `diffsynth/pipelines/wan_video_new.py` | `model_fn_wan_video`；pre-DiT/ hidden_sequence 集成；counterfactual mask 目前只覆盖 history |
| `videopress_framework/scripts/run_official_navsim_press.py` | 官方 NAVSIM runner；当前 `--domain` 只允许 history；future 实验入口需要改这里 |
| `videopress_framework/scripts/run_full_compression_suite.py` | synthetic 19-method matrix；`method_specs(domain=...)` 已可在框架层传 `future_video` |

---

## 3. 环境、数据、Checkpoint

### 3.1 Python / 硬件

- 推荐 Python：`/home/cpj/miniconda3/envs/DriveVA/bin/python`（Python 3.10，CUDA PyTorch）。
- 另一等价路径：`/mnt/nvme/chenpeijian/miniconda3/envs/DriveVA/bin/python`。
- 分布式运行使用 `torch.distributed.run`；历史 full run 常用 6 卡（logical GPU 0–5），
  单卡 smoke 用 `CUDA_VISIBLE_DEVICES=0`。
- **并发上限（2026-09-20 23:15 用户再次收紧）**：当前仅允许同时占用 **2 张 GPU**（建议 0/1，但 0/1 若被他人占用则用最空闲的两张）。所有 GPU
  任务必须通过 tmux 队列 `outputs/future_oracle_queue_v3_20260918/run_queue.sh` 排队执行，
  启动前检查 `nvidia-smi` 与 `ps`，不允许直接抢占 GPU。
- 绘图前设 `MPLCONFIGDIR=/tmp/driveva_mpl`，避免 matplotlib 写 home 失败。

### 3.2 模型与 checkpoint

```text
models -> /mnt/nvme/chenpeijian/autodrive/DriveVA/models
checkpoints/pdms90_9.safetensors
  -> /mnt/nvme/chenpeijian/autodrive/DriveVA/checkpoints/pdms90_9.safetensors
```

Wan2.2-TI2V-5B 默认在 `models/Wan-AI/Wan2.2-TI2V-5B`。

### 3.3 官方 NAVSIM 主协议（7,876 场景）

来自 runner 内置 `EVAL_PROTOCOLS["navtest-7876"]`：

- log：`/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/openscene-v1.1/meta_datas/test`
- sensor blobs：`/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/openscene-v1.1/sensor_blobs/test`
- metric cache：`/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_full`
- scene filter：`examples/wanvideo/driveva_infer/navsim_scene_filters/navtest.yaml`
- 基线：NoPress `0.9098390680735604`；deployed learned threshold40 `0.9096916412430458`
- 保护条件：必须使用 same-scene guard；当前 guarded 集合 7,876，全部有 cache。

次协议 **split-test-1920**：

- metadata：`videopress_framework/outputs/navsim_official_test_repaired/metadata`
- metric cache：`/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_split_test`
- scene filter：`.../official_test_repaired_1920_scene_filter.yaml`
- 绝对 PDM 比 7,876 低约 1 分，只用于 paired delta。

### 3.4 Selector 训练数据

- 独立训练场景 manifest：`videopress_framework/outputs/navsim_split_audit/train_manifest.jsonl`（3,768 场景）。
- 禁止出现在训练中的测试 manifest：`.../test_manifest.jsonl`（1,920 场景）。
- 当前最佳 selector capture split：`videopress_framework/outputs/selector_capture_split_20260910/`
  - train：`selector_train_manifest.jsonl`（3,190 场景）
  - calibration：`selector_calibration_manifest.jsonl`（578 场景）
  - split report：`split_report.json`，train/val/test 无 scene overlap。
- 训练 metadata：`videopress_framework/outputs/navsim_split_audit/metadata/train`
- 训练 sensor blobs：`/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/extra_trainval_32/openscene-v1.1/sensor_blobs/trainval`

### 3.5 当前 DriveVA eval 的 token 布局（默认 480×832, history=5, future=8）

- VAE latent temporal length：`(total_frames - 1) // 4 + 1`。
- `total_frames = 5 + 8 = 13`，latent frames `f = 4`。
- `num_cond_latents = 2`（history）。
- future latent frames：`4 - 2 = 2`。
- 每个 latent 在当前配置下是 **390 tokens**：
  - history：`2 × 390 = 780` candidate；
  - future：`2 × 390 = 780` candidate；
  - video total：`4 × 390 = 1560`；
  - 再加 trajectory/action token 后总序列长度约 `1569`。
- 因此 future 与 history 的候选规模相同，这也是“future 能否复用 history select 参数”的直接背景。

---

## 4. 已验证工作、结果与结论

### 2026-08-18 — 领域定位

- 场景状态稀疏：SparseBEV / Sparse4D / SparseDrive；
- 动作搜索稀疏：SparseDriveV2 的 path-speed decomposition + coarse-to-fine；
- World Model：Drive-WM（图像/视频未来）、OccWorld（BEV/occupancy）、LAW（latent +
  trajectory）、WoTE（多轨迹 + action token + future BEV latent）。
- SparseWorld：VAD-Tiny L2 0.78→0.59 m，碰撞率 0.38%→0.24%；SparseDrive-S 0.08%→0.05%；
  4 帧生成 70 ms / 4,397 MB，对比 Drive-OccWorld 398 ms / 20,581 MB。
- 结论：在 SparseDriveV2 上叠加 Latent WM 可行，但会与 LAW/WoTE 等思路重合；
  对本项目更重要的是 **DriveVA video/action token 压缩**，不是重做 World Model。

### 2026-08-22 — 稀疏性、选择有效性

- action→video attention 很集中：最高 20% history token 承载约 77.6% attention mass。
- 但**集中不等于选择有效**：32 场景 Attention Top-K 平均 PDM `0.833`，五个 Random
  seeds 均值 `0.851`；Attention Top-K 没有相对随机选择的优势。
- Gradient×Input：
  - 32 场景 Top-156：PDM retention `0.9997`；
  - 扩到 64 场景后 retention 降到 `0.940`，五个 Random seeds 平均 `0.919`；
  - Gradient 比 Random 平均高 `0.020 PDM`，但不稳定。
- Object-region mean-8 attention 富集约 `1.85×`，说明 attention 与语义有关。
- 但 Attention 与 planning attribution 一致性弱：Spearman 相关 `0.224`；
  Top-156 overlap/K `0.498`（随机期望 `0.40`）。
- 结论：video token 存在大量功能等价子集，随机保留已能保留大部分 planning 性能；
  当时没有找到稳定、可泛化、明显优于随机的 hard selector。下一步转向 learned latent
  bottleneck / 时序压缩。相关方向：DrivoR、EfficientVLA、DynamicViT。

### 2026-09-08 — 物理压缩、双 latent retention、数据集清理

1. `physical_attention_vnorm_kv_prune` 的解释：
   - 第 15/16 层 token 语义分割性较强；
   - 重要性 = action attention × value norm，即 `score_i = alpha_i * ||V_i||_2`；
   - 按 Top-K 真正删除未保留的 history K/V。
2. 数据：
   - 下载 3,768 个独立场景用于 selector 训练；
   - 原 official test 1,920 个可用场景保持评测；
   - 对冗余 LiDAR 数据做分离清理、场景去重。
3. 历史 latent retention：
   - 原先两个 VAE history latent 的保留位置固定，改为可调节；
   - 六种精确保留策略。
   - NoPress baseline PDM `0.909839`；
   - 六种 quota 的 PDM 都回到基附近，只保留约 25% history candidate 未观察到明显退化；
   - 单层物理裁剪把 K/V 从 `1569` 降到约 `1179/984`，但端到端延迟仍增加：
     单层节省不足以抵消评分和 gather 开销。
4. 结论：跨层持续压缩比单层 K/V prune 更有价值；这个时期开始做 cross-layer persistent 模块。

### 2026-09-10 — 跨层持续压缩首次成型

核心变化：

- 旧：源层单次裁 K/V，后续层仍处理完整 residual；
- 新：Layer 16 评分选一次 → residual/RoPE/t_mod `1569 → 1325` →
  Layer 17–29 共 13 层跑短序列 → Head 前恢复 `1569`。

90 场景两轮结果（report）：

| 方法 | history 保留 | PDM | 延迟 |
|---|---|---|---|
| NoPress | 780/780 | 0.909839 | 589.74–591.92 ms |
| Layer 16 Joint50 | 390/780 | 0.907916（−0.001923） | 553.88 ms（−6.08%） |
| Layer 15 Joint75 | 585/780 | 0.910399（+0.000560） | 585.11 ms（−0.78%） |
| Attention-VNorm adaptive | ~558/780 | 0.909944（+0.000105） | 578.64 ms（−2.24%） |

结论：

- Joint50 用显著质量损失换 6.08% 加速；Joint75 保质量但只有 0.78% 加速；
  动态预算在两者间取得较好折中，但无法超过 NoPress。
- 更深层 scorer 计算更重，均未超过轻量 Attention-VNorm。
- 开始怀疑一个结构前提：Layer 15/16 只选一次，却让同一集合服务后续 13–14 层；
  如果 token importance 随深度/时间变化，再准确的单层排序也会失效。

### 2026-09-14 — 时间与跨层连续性分析

方法核心：对每个 scene、history latent、spatial tile 做反事实删除，用预测轨迹变化
定义重要性 `u_{s,r,i}`；再统计排序一致性与 Top-K overlap。

结果：

| 检验 | 统计量 | 实测 | 随机基线 | 结论 |
|---|---|---|---|---|
| 相邻 latent 排序迁移（192×12×2） | pairwise concordance C | 0.5318 | 0.4999 | +6.4%，只略高于随机 |
| 前一 latent Top-6 覆盖后一 latent | importance mass Q6 | 0.5129 | 0.5000 | +2.6% |
| 静态位置先验（2-fold CV） | 固定 tile 排序 | 0.5659 | 0.5000 | 比动态迁移更强 |
| Layer 15→20 集合迁移 | Top-K overlap | 0.5224 | 0.5000 | +4.5% |
| Layer 15→20 排序迁移 | rank AUC | 0.5269 | 0.5000 | +5.4% |
| Layer 15→25 集合迁移 | Top-K overlap | 0.5061 | 0.5000 | +1.2%，接近随机 |
| Layer 15→25 排序迁移 | rank AUC | 0.5111 | 0.5000 | +2.2% |

结论：

- 时间连续性弱于静态位置先验；Layer 15 选择到 Layer 25 已与随机不可区分。
- 原因：token 是固定 spatial patch，不是有身份的目标实例；两段 history latent 之间有
  自车运动、遮挡和 VAE temporal compression，不能把同一索引理解为同一世界对象。
- 跨层弱说明 DiT 会不断重写 token 表征：Layer 15 的“重要”只描述该层当前计算需求，
  不代表深层还一定需要同一位置。
- **重要解释转变**：Layer 15/16 以后可删部分 token，不是找到了永久无效 token，而是
  模型已在前半段完成语义混合，部分信息被写入其他 token；Press 利用的是已产生的表示冗余，
  不是替模型完成感知选择。
- 尚未排除：质量问题可能不是“选不准”，而是“直接删除太粗糙”。因此下一阶段比较
  Pre-DiT 删除、merge、可学习 register。

### 2026-09-16 — Pre-DiT 删除、Greedy Merge、RegisterBottleneck

| 方法 | 规模 | PDM | 延迟/判断 |
|---|---|---|---|
| NoPress | 32 / 256 / 7,876 | 0.9293 / 0.906841 / 0.909839 | 603.6 ms / 593.46 ms |
| Learned prune, K=195 | 32 | 0.8461（−9.0%） | 556.9 ms（−7.7%），但比同预算 random 更差 |
| Random prune, K=195 | 256 | 0.884331（−2.5%） | 删除损失明显 |
| Greedy cosine merge, K=195 | 256 / 7,876 | 0.905666 / 0.900016 | 256：615.62 ms（+3.7%）；全量 −1.08% |
| RegisterBottleneck + rank-32 LoRA | 256 | 约 0.634（−30.1%） | 增加 K、训练步数均未恢复，失败 |

全量结论：

- Greedy merge 的全量下降不是均匀退化：7,581 个非极端场景平均 Δ `+0.003528`，
  但 295 个极端翻转拖垮总体。
- NoPress/merge 零分场景 `243/350`，其中新增 199、救回 92；
  新增零分全部首先表现为 `ego_progress=0`。**小面板接近 NoPress 不代表尾部安全。**
- Pre-DiT learned prune 失败原因：压缩发生在语义形成之前；selector 读取的仍是接近
  VAE patch 的低层特征，却要提前判断 30 层之后的规划作用。
- Greedy merge 只保证视觉邻近，不能保证交通控制、路线边界等少量证据被保留。
- RegisterBottleneck 问题：全局 cross-attention 容易把罕见、局部、决定路线推进的信息
  平均掉；主要矛盾是聚合归纳偏置与规划需求不匹配，不是训练步数或容量。
- 至此三种解释都被排除：不是更强 scorer、不是删除改 merge、也不是简单增加可学习容量即可。
- 结论：Press 不再承担提高能力的任务，只利用模型已有冗余实现近无损加速；
  能力提升应与 Press 解耦。

### 2026-09-17 — 当前最佳结果总览

7,876 场景主协议：

| 方法 | 规模 | PDM | 延迟 | 判断 |
|---|---|---|---|---|
| NoPress | 7,876 | 0.909839 | 589.74—591.92 ms | 基线 |
| Layer 15 Joint75 | 7,876 | 0.910399（+0.000560） | 585.11 ms（−0.78%） | PDM 差异不显著，几乎无加速 |
| Attention-VNorm 分档变化 | 7,876 | 0.909944（+0.000105） | 578.64 ms（−2.24%） | 无训练、近无损加速 |
| Learned 双 latent Press 动态阈值 | 7,876 | **0.911143（+0.001304）** | 568.40 / 567.69 ms（−2.84% / −3.41%） | 当前最佳 Press，质量近无损 |
| Layer 0 greedy merge | 256；7,876 | 0.905666；0.900016 | 615.62 ms（+3.7%） | 小面板近无损，全量下降 |
| RegisterBottleneck | 256 | 约 0.634（−30.1%） | — | 失败 |

当前最佳 learned Press 的精确配置（来自 `config.json`）：

```text
domain: history
injection_point: self_attn_kv
scorer: learned_planning_selector
  layer: 15
  feature_layer: 15
  checkpoint: videopress_framework/outputs/magnitude_teacher_capture_train3190_2win_20260913/step-3190.safetensors
selector: history_threshold
  thresholds: [0.05, 0.40]   # oldest, newest
operator: kv_prune
budget: ratio 1.0, reference=eligible  # 阈值决定数量，budget 只是安全上限
cross_layer_persistence: enabled=true, mode=hidden_sequence
```

产物：

- 最佳 full run suite root：
  `videopress_framework/outputs/audit_dynamic_plan_20260913/capture_dynamic_navtest7876_seed0_rerun01/`
- 方法目录：
  `.../round01/physical_learned_planning_selector_history_threshold_hidden_persistent_layer_15/`
- paired vs NoPress：
  `.../paired_vs_no_press_report.json`
  - PDM `0.9111427046036312` vs `0.9098390680735642`
  - Δ `+0.001303636530068348`，95% CI `[-0.000776, +0.003399]`，**跨 0**
  - 极端翻转 73 个；非极端 Δ `+0.000333`，CI 仍跨 0
  - reference zero count 243，candidate zero count 233，rescued 41，introduced 31
- 方法统计表：
  `.../statistics/method_statistics.md`
- 训练 checkpoint：
  `videopress_framework/outputs/magnitude_teacher_capture_train3190_2win_20260913/step-3190.safetensors`

注意：

- 当前最佳结果的提升 CI 跨 0，因此正确表述是 **近无损加速 / 不显著正向**，
  不是“Press 提高了 PDM”。
- 延迟受配对 batch / 评测负载影响，报告中的 589.74—591.92 ms 等是 paired 同批数据；
  引用时必须写清楚比较基准。

---

### 2026-09-17 — Future domain 打通 + 64-scene 零样本迁移 POC

**代码侧 Phase F0（已实现，195 tests pass）**：

- 新增 `FutureLatentDomain`：`future_latent_0` = 最近未来 latent（storage index
  `num_cond_latents + 0`），`future_latent_1` = 下一个未来 latent；顺序固定为 storage /
  near-to-far。
- 官方 runner `--domain` 开放 `future_video`、`future_latent_0`、`future_latent_1`；
  `--persistent-selector` 新增 `future_threshold` / `future_quota`，并新增
  `--per-future-latent-thresholds` / `--per-future-latent-keep-ratios`。
- `--persistent-scorer` 支持 persistent `random` control；random 不再收到错误的
  `action_mode` 参数。
- learned selector `_positions()` 支持 future domain 的两种模式：
  - `storage`: `t=2,3`；
  - `history_compatible`: future near/far 重映射为 `t=0,1`。
  该模式由 `--persistent-future-position-mode` 控制。
- Budget 支持 `each_future` / `per_future_latent`；metadata 增加 future latent 保留统计。
- history-only retention policy 遇到 future domain 显式报错，不再静默改成 history。

**POC 设置**：官方 `navtest-7876` 前 64 个窗口，`--max-eval-tokens 64`，单 seed，
Layer 15 + `hidden_sequence` + `kv_prune`，learned checkpoint 为当前 history best
`outputs/magnitude_teacher_capture_train3190_2win_20260913/step-3190.safetensors`。
这是 truncated POC，不能当全量 PDM 结论。

**POC 主要结果（ΔPDM 为 paired bootstrap 95% CI）**：

| Arm | K/780 | ΔPDM | 95% CI | 判断 |
|---|---:|---:|---:|---|
| `aa50` | 390 | −0.0392 | [−0.0853, −0.0069] | 50% future hard prune 明显掉点 |
| `random50` | 390 | −0.0547 | [−0.1068, −0.0134] | 同预算 random 更差 |
| `aa68` | 535 | −0.0266 | [−0.0710, +0.0039] | 68.6% 保留仍不 near-lossless |
| `random68` | 535 | −0.0331 | [−0.0801, −0.0006] | 同预算 random 仍差 |
| `learned_hc_thr` | 536.9 | −0.0651 | [−0.1148, −0.0258] | history selector 零样本失败 |
| `learned_storage_thr` | 531.7 | −0.0861 | [−0.1442, −0.0389] | storage position 更差 |
| `learned_hc_topk50` | 390 | −0.0677 | [−0.1267, −0.0209] | K 匹配 random 仍更差 |
| `learned_storage_topk50` | 390 | −0.0670 | [−0.1256, −0.0205] | 同上 |

关键 pairwise：

- `aa68 − random68 = +0.0065`，CI [−0.0529, +0.0666]；attention 不显著优于 random。
- `learned_hc_thr − random68 = −0.0320`，CI [−0.0786, +0.0139]；history selector 零样本没有优势。
- `learned_hc_thr − learned_storage_thr = +0.0210`，CI [−0.0005, +0.0561]；
  `history_compatible` 比真实 storage coordinate 好，位置 OOD 是因素之一。

**产物**：

- 报告：`videopress_framework/outputs/future_poc64_report_20260917.md`
- 聚合 JSON：`videopress_framework/outputs/future_poc64_summary_20260918.json`
- 原始输出：`videopress_framework/outputs/future_poc64_*_v2_20260918*/`
- 单场景 smoke：`outputs/future_token_smoke_20260918/`、
  `outputs/future_learned_hc_smoke_20260918/`。

**结论（POC，需全量复核）**：

- history 的“Layer 15 选一次 + hidden_sequence”机制在 future 上物理可运行，但不 near-lossless；
- history-trained learned selector / 同参数阈值不能零样本迁移，且在匹配 K 下不优于 random；
- future 不存在“肉眼可见的 free lunch”；进入 F3 全量训练前必须先做 F1 future oracle/上界。

---

### 2026-09-17 — History-guided future compression（用户新主线）

**动机**：future token 早期仍是纯噪声，直接对 future 打分接近随机；用户提出若 future
同一位置编码的 history token 被保留，则 future 同位置也更值得保留，从而把压缩近似乘 2。

**已实现代码**：

- `HistoryGuidedFutureSelector`：在 `all_video` 上先用 `[oldest,newest]` history 阈值选
  history，再把历史局部 spatial mask 映射到 future；支持 `same_latent`、
  `reverse_latent`、`nearest_history`、`oldest_history`、`union_history`、
  `intersection_history`、`majority_history`。
- `LearnedPlanningSelectorScorer(all_video_history_only=True)`：只在 history 子域跑
  learned network，future 分数置零，确保 future 噪声不参与选择。
- runner：`--history-guided-future-mapping`、`--history-guided-layer`、
  `--history-guided-thresholds`。

**1024-scene POC**（4-rank，`--max-eval-tokens 256`，每 rank 256 个不同场景，共 1024
个 paired 场景；NoPress PDM `0.911205`，接近全量 `0.909839`；仍不是 full 7,876）：

| Arm | PDM | ΔPDM vs NoPress | 95% CI | K/1560 | hidden | latency | zero candidate/NoPress |
|---|---:|---:|---:|---:|---:|---:|---:|
| NoPress | 0.9112 | — | — | — | 1569 | 586.4 | 30 / 30 |
| history best (full-run matched) | 0.9073 | −0.0039 | [−0.0082, −0.0005] | 489.6 | 1291.9 | 569.3 | 34 / 30 |
| `union_history` | 0.8991 | −0.0121 | [−0.0201, −0.0043] | 1148.9 | 1197.7 | 556.4 | 44 / 30 |
| `same_latent` | 0.8832 | −0.0280 | [−0.0395, −0.0169] | 988.9 | 1038.1 | 537.6 | 49 / 30 |

**结论**：

- 用户直觉作为 selection prior 成立一部分：复制 history 位置比在 future 噪声上选更
  可控，且 `union_history` 比直接 future hard select 的 POC 曲线更好；
- 但“同位置复制 => 压缩乘 2”在这一版阈值下 **不是 near-lossless**：
  `same_latent` 额外丢约 2.8 PDM 点，`union_history` 额外丢约 1.2 点；
- 相对 history best，同一组 1024 场景下 `union_history` 多省约 13 ms、少约 94 个
  hidden token，但多掉约 0.8 PDM 点；
- 因此下一步应做 history-guided threshold / future keep-ratio 的 frontier sweep，
  而不是直接训练 future selector 或全量部署。

产物：`outputs/history_guided_future_poc1024_report_20260917.md`、
`outputs/hgf_poc64_*_20260918/`、`outputs/hgf_poc256_union_history_20260918/`、
`outputs/hgf_poc1024_same_latent_20260918/`。

---

### 2026-09-17 — history-guided future frontier sweep（common 512 scenes）

在 `history_guided_future` 机制上做了 threshold / future keep-ratio sweep，并加入其他路线
对比。所有行用同一组 512 paired scenes（NoPress PDM `0.911362`；POC，非 full 7,876）。

| 路线 | ΔPDM | 95% CI | hidden length | latency ms | 说明 |
|---|---:|---:|---:|---:|---|
| `union_history` `[0.02,0.40]` | −0.0004 | [−0.0074,+0.0064] | 1330.8 | 578.3 | 近无损但比 history-only 更慢、压缩更少 |
| `union_history` `[0.03,0.40]` | −0.0105 | [−0.0219,−0.0001] | 1296.2 | 568.9 | 同长度下不如 history-only |
| `union_history` `[0.04,0.40]` | −0.0093 | [−0.0203,+0.0006] | 1249.2 | 564.5 | 有加速但 CI 上界贴 0 |
| `union_history` `[0.05,0.40]` | −0.0158 | [−0.0281,−0.0050] | 1198.1 | 557.0 | 当前 guided 基准 |
| `union_history` cap 0.85 | −0.0162 | [−0.0295,−0.0042] | 1165.2 | 555.1 | cap 开始明显掉点 |
| `union_history` cap 0.80 | −0.0201 | [−0.0344,−0.0069] | 1135.7 | 552.8 | 更差 |
| `same_latent` | −0.0321 | [−0.0486,−0.0164] | 1038.5 | 538.1 | 压缩乘 2 假设失败 |
| history-only best | −0.0060 | [−0.0130,−0.0004] | 1292.2 | 569.7* | *full-run latency，matched PDM |
| future direct `action_attention_vnorm` topk0.75 | −0.0276 | [−0.0455,−0.0113] | 1374.0 | 572.4 | 直接 future selector 更差 |
| future direct `action_attention_vnorm` adaptive | −0.0094 | [−0.0208,+0.0013] | 1507.3 | 591.4 | 几乎不压缩且更慢 |

结论：

- future keep-ratio cap 越小越差；
- 唯一近无损的 guided 点 `[0.02,0.40]` 压缩比 history-only best 少且更慢；
- 在相近 physical length 下，guided future 仍比 history-only best 多掉约 0.5 PDM 点；
- 因此当前 mask-copy history-guided future **没有找到 near-lossless 的乘 2 工作点**。

产物：`outputs/history_guided_future_frontier_512_report_20260917.md`。

---

### 2026-09-17 — Future subset oracle 上界初测

方法：官方 runner `future_video` + Layer15 hidden_sequence + random masks；每个 scene
从 N 个随机 mask 的 official PDM 里取 max，作为“随机子集 oracle 上界”。POC 64 scenes，
NoPress PDM `0.940644`；该 oracle 有明显选择偏差，只能用于判断 headroom。

| future keep | K/780 | N | single-seed ΔPDM | random-oracle ΔPDM | oracle 95% CI | oracle PDM |
|---|---:|---:|---:|---:|---:|---:|
| 0.500 | 390 | 16 | −0.0671 | **+0.0026** | [−0.0038,+0.0084] | 0.9432 |
| 0.375 | 292 | 16 | −0.1092 | −0.0209 | [−0.0300,−0.0124] | 0.9197 |
| 0.250 | 195 | 8 | −0.1608 | −0.0709 | [−0.0983,−0.0485] | 0.8697 |

结论：

- 50% future keep 的 oracle 上界与 NoPress 不可区分，说明该预算下存在大量 planning
  冗余子集，这是目前最强的 positive signal；
- 37.5% oracle 仍掉 0.0209，但比 single-seed random 好很多，不能排除更强 teacher
  能找到更好子集；
- 25% oracle 明显掉点，冗余不是无限；
- 随机 seed 不是部署方法，下一步必须做 counterfactual tile / planning-gradient oracle。

产物：`outputs/oracle_future_subset_upper_bound_20260917.md`、
`outputs/oracle_random_k50_seed1001..1016_20260918/` 等。

---

### 2026-09-18 — future counterfactual tile oracle 完成：失败结论

tmux 队列 `driveva_oracle` 已完成 matrix128 + matrix512 及三档 oracle eval（每阶段最多 2 GPU）。
结果如下：

| Panel | keep | ΔPDM vs NoPress | 95% CI |
|---|---:|---:|---:|
| 128 scenes | 0.500 | −0.0610 | [−0.1099,−0.0170] |
| 128 scenes | 0.375 | −0.1080 | [−0.1583,−0.0631] |
| 128 scenes | 0.250 | −0.1082 | [−0.1529,−0.0681] |
| 512 scenes | 0.500 | −0.0424 | [−0.0637,−0.0220] |
| 512 scenes | 0.375 | −0.0891 | [−0.1129,−0.0665] |
| 512 scenes | 0.250 | −0.1129 | [−0.1385,−0.0882] |

matrix512 的 tile-level 统计：mean harm 约 `[-0.004,+0.009]`，SE 约 `0.003–0.006`，
`harm>0` 比例 `0.23–0.45`。说明单次 PDM leave-one-tile-out 信号接近噪声，
per-scene tile ranking 不可靠。

结论：

- 当前 counterfactual tile oracle **不是有效 oracle**，在 keep0.5 的 512 面板上 ΔPDM
  `−0.0424`，明显差于 history-guided union（约 `−0.012`）和 attention-vnorm adaptive
  （约 `−0.009`）；
- tile 重要性不满足可加性，单 tile PDM 差异无法外推到组合子集；
- 该结果既不能证明 future 可压缩，也不能证明不可压缩，只证明此构造方法失败；
- 下一步应使用 trajectory displacement / multi-replay、set-level greedy/beam oracle，
  并补 matched random control。

产物：`outputs/future_oracle_counterfactual_conclusion_20260918.md` 及
`outputs/future_oracle_*_20260918/`。

---

### 2026-09-20 — token-level set-level oracle 全部完成：1024 场景判决性否定结果

**最终验证面板**：`--max-eval-tokens 256` × 4 rank = **1024 个不同场景**（不是 4 个 seed × 256，
实测四个 v3 "seed" 面板是**同一组 256 场景**）；Layer 15 + `hidden_sequence` + `kv_prune`，
keep 0.50（K=390/780）。NoPress 基线 **0.911205**（与历史 1024-scene POC 基线一致，交叉校验通过）。
运行 19:25:32 → 20:35:33，报告
`outputs/future_token_set_oracle_verify_1024_20260920/run/oracle_search_report.{json,md}`。

| arm | K | ΔPDM vs NoPress | 95% CI | 在随机 band 中的位置 | zero cand/base | extreme |
|---|---:|---:|---|---|---:|---:|
| 随机 band（15 个 mask） | 390 | min −0.0557 / mean **−0.0483** / max −0.0413（sd 0.0036） | — | — | — | — |
| stage2 independent top-K 组合 | 390 | −0.0534 | [−0.0686,−0.0386] | 只胜过 **13%** 随机臂 | 80/30 | 71 |
| stage3 greedy 搜索 mask | 270 | −0.0726 | [−0.0882,−0.0575] | 胜过 **0%** 随机臂（**预算不匹配**，待公平复核） | 81/30 | 73 |

**结论（verified）**：

1. **在 1024 场景上，future keep 50% 的随机 hard prune 就要掉 `−0.048`**，远不是近无损；
   此前"50% 预算存在大量冗余"的乐观结论建立在 64 场景（best-of-16 oracle +0.0026）与
   256 场景（random −0.0142）之上，**在更大面板上不成立**（`[已被新结果推翻]`）。
2. **没有任何被搜索/构造出来的 mask 优于 matched random**：预算匹配的 independent top-K
   组合只排在随机 band 的 13 分位（paired vs best random `−0.0121`，CI `[−0.0273,+0.0027]`）；
   greedy 搜索出的 270-token mask 在 1024 场景上比所有 15 个随机臂都差
   （paired vs best random `−0.0313`，CI `[−0.0471,−0.0158]`）。
3. **小面板搜索会过拟合**：stage3 在 64 场景上选出 270 token 的 mask，面板内 +0.0189 且优于
   best-of-16 随机（+0.0092）；到 1024 场景变成 −0.0726、输给全部随机臂。64 场景的
   配对分辨率只有 ±0.025，搜索只是在拟合噪声。**不要用小面板搜索来选择可部署 pattern。**
4. **zero-score 尾部恶化**：候选引入的新零分场景 71–81 个（基线仅 30 个），与 zero-tail 一致。
5. **采样噪声标定**（v3 matrix：同一 24 个 mask × 4 个 `--sample-seed`，256 场景）：
   绝对 NoPress panel 均值随种子 `0.880333/0.904601/0.911602/0.901051`（极差 **0.0313**）；
   同一固定 mask 的**配对** ΔPDM 跨种子 sd 仅 **0.0067**。→ 绝对 PDM 不能跨采样种子比较；
   配对 ΔPDM 才有效；功效主要靠增加**场景数**（不是种子数）。


**后续补测完成（21:24→22:09，同一 1024 场景面板，`outputs/future_token_set_oracle_followup_20260920/`）**：

keep-ratio frontier（matched-budget **随机**控制，NoPress `0.911205`）：

| keep | K | ΔPDM（各臂 CI） | zero cand/base | per-scene best-of-N oracle |
|---|---:|---|---:|---|
| 0.346 | 270 | −0.0952…−0.1018，CI 上界均 ≤ −0.0767 | 65–77 / 30 | **−0.0324**（仍然很差） |
| 0.500 | 390 | mean −0.0483（15 臂） | ~? | （256 场景上 +0.059，选择偏差） |
| 0.750 | 585 | −0.0204…−0.0234，CI 上界 −0.0102…−0.0132 | 51–55 / 30 | +0.0079 |
| 0.875 | 682 | −0.0068…−0.0149，CI 上界 −0.0002…−0.0077 | 37–45 / 30 | +0.0050 |

**预算匹配复核（重要修正）**：stage3 greedy 搜索出的 **270-token** mask 与 **270-token**
随机 band 对比：searched `−0.0726` CI `[−0.0882,−0.0575]`，随机 band `−0.1018…−0.0913`
（mean `−0.0978`），**searched 胜过 100% 随机臂，paired `+0.0187` CI `[+0.0029,+0.0339]`
（不跨 0）**。
→ 之前"没有任何 mask 优于 matched random"的说法**是 K=270 vs K=390 预算不匹配造成的假象，予以更正**：
**selection 在匹配预算下确实有信号（+0.019），但绝对水平只有 −0.073，救不回来。**

**最终结论（本阶段）**：

1. **问题答案是否定的**：不存在使 future hard prune 近无损的 token-level set-level oracle。
   keep 0.5 上任何构造/搜索的 mask 都在 −0.05 量级；即使 keep 0.875（只丢 12.5% future
   token ≈ 6.2% 序列长度）随机控制最好也只有 `−0.0068`，仍比 `−0.002` 的 near-lossless
   门槛差 3 倍，且所有臂 CI 均不跨 0（显著变差）。
2. **selection 信号真实但太弱**：K=270 时 searched 比随机好 `+0.019`（CI 不跨 0），
   说明 future token 重要性不是纯噪声；但预算太小时绝对损失无法挽回。
3. **零分尾部随压缩加深单调恶化**：30 → 37–45（keep .875）→ 51–55（.75）→ 65–77（.346）。
4. **50% future keep 不存在"大量冗余"**：1024 场景随机控制 `−0.0483`，推翻 64/256 场景
   小面板给出的乐观读数。
5. **F3（训练 future selector）取消**；**future hard prune 停止**；保留已部署的 history press
   （7,876：`0.911143`，延迟 −2.84%/−3.41%，CI 跨 0）。
6. 未测但已知：所有结论只覆盖 trajectory PDM；`hidden_sequence` 在 Head 前把 dropped token
   置零，future video decode 质量必然受损，不能用 PDM 近似代表。

**收尾检查（keep 0.875 上的可部署 scorer，22:25→22:30，1024 场景）**：
`action_attention_vnorm` top-k → ΔPDM **`−0.0133`** CI `[−0.0211,−0.0058]`，
只胜过 33% 的随机臂，paired vs best random `−0.0065` `[−0.0155,+0.0024]`，
zero `44/30`。近无损门槛（CI 下界 > `−0.002`）**未通过**，
在唯一近中性档位（−6.2% 序列长度）上**可部署 scorer 仍不如随机**。
→ 最后一个漏洞关闭。产物：`outputs/future_keep0875_avnorm_1024_20260920/`。
结论报告：`outputs/future_token_level_oracle_conclusion_20260920.md`。

**下一步选项（推荐 #1）**：
1. 关闭这条线，写 F1/F4 结论报告；能力侧转 structured attention / training-time
   bottleneck / merge / quantization；
2. 若仍想确认唯一近中性档位：keep 0.875 上跑一次**可部署** `action_attention_vnorm`
   scorer（1 臂，1024 场景，约 5 min），看它能否超过随机 band（现随机最好 −0.0068）；
3. 若目标是 history+future 联合 press：在 1024 场景上测联合工作点（约 15 min）。

**下一步（已启动）**：`outputs/future_token_set_oracle_followup_20260920/run_followup.sh`
（21:24 起，约 50 min）补两件事：
1. **预算匹配复核**：270-token 随机 band（5 臂），公平判定 greedy mask；
2. **keep-ratio frontier**：同一 1024 场景面板上随机控制在 keep 0.75 / 0.875 的水平，
   回答"future token 到底能压多少"。

若 keep 0.75 仍远非近无损 → **彻底放弃 future hard prune**，转 structured attention /
training-time bottleneck / merge；并且 **F3（训练 future selector）直接取消**：
oracle 上界都不如随机，没有可蒸馏的信号。

## 5. 当前框架能力矩阵

| 能力 | 状态 | 说明 |
|---|---|---|
| token layout / domain | 已实现 | `history_video`、`future_video`、`history_latent_i`、`future_latent_i` 全部存在 |
| causal / physical 协议 | 已实现 | physical 支持 `SELF_ATTN_KV` + `kv_prune`、`BLOCK_INPUT` + `hidden_prune` |
| cross-layer persistent K/V | 已实现 | source layer 选一次，后续层 gather 新 K/V 或复用 mapping |
| hidden_sequence | 已实现 | source layer 后缩短 residual/RoPE/t_mod；Head 前 restore 为 zeros |
| Attention / Attention-VNorm | 已实现 | 可作用于任意 video domain，包括 future_video |
| Adaptive mass dynamic budget | 已实现 | 通用 selector，可作用于 future domain |
| Learned planning selector | 部署已支持 future；训练仍只支持 history | 部署 `_positions()` 支持 storage / history_compatible future 模式；训练 capture/mask/teacher 未扩展 |
| history 双 latent threshold/quota | 已实现 | `history_threshold`、`history_quota` 只对 history |
| future 双 latent threshold/quota | 已实现 | `future_threshold`、`future_quota`，顺序固定 near-to-far |
| history-guided future mask transfer | 已实现 | `history_guided_future` selector、`all_video_history_only` scorer、`future_keep_ratio` cap、runner CLI；1024/512 POC + frontier sweep 未找到 near-lossless 乘 2 工作点 |
| future domain 官方 runner | **已开放** | `--domain future_video/future_latent_0/future_latent_1` 可用 |
| future latent 单独 domain/budget | **已实现** | `future_latent_i`、`each_future` reference 已加入并有测试 |
| future online selector 训练 | **未实现** | `capture_history_tokens` / `history_token_mask` / `counterfactual_latent_index` 仍只覆盖 history |
| future oracle/上界分析 | random-mask / tile / token-level set-level 三种 oracle 均已完成，**全部否定** | 1024 场景 keep0.5：随机 band `−0.0483`，搜索 mask 不优于随机（13 分位 / 输给全部臂）；tile `combined keep0.5` `−0.0182`；下一步只测 keep-ratio frontier |
| **token-level set-level future oracle** | **已实现并跑完（代码 + 39 单测）；1024 场景验证为否定结果** | `oracle_future_token_mask` selector、`--future-oracle-token-mask-json(s)`、`videopress/oracle/`（token 分组 + set-level greedy/beam/random）、`scripts/search_future_token_set_oracle.py` |
| future physical smoke | 已跑通 official single scene + 64-scene POC | `outputs/future_token_smoke_20260918/`、`outputs/future_poc64_report_20260917.md` |

---

## 6. 当前工作区未提交状态（2026-09-20 23:50 更新）

`git status`：`main @ d6a423d`，**与 `origin/main` 同步，工作区 clean**；唯一 untracked 是
排除项 `videopress_framework/scripts/pre_dit_gpu_smoke.py`（按 §9.6 不进 git）。
token-level oracle 的全部代码、测试、文档与联合 frontier 结果**已提交并推送**。

### 6.1 已提交内容（本阶段）

| commit | 内容 |
|---|---|
| `87376e8` | token-level set-level oracle 代码（`videopress/oracle/`、selector、runner CLI、搜索驱动）+ 39 单测 |
| `9549593` | README / AGENTS.md 文档同步 |
| `a82a849` | 收尾检查 + 结论报告索引 |
| `d6a423d` | runner 新增 `--domain all_video`（联合 matched-K 对照必需）+ 单测 + 联合结果与 2-GPU 资源规则 |

### 6.2 未提交 / 排除项

```text
?? videopress_framework/scripts/pre_dit_gpu_smoke.py   # 排除项，GPU smoke 临时脚本
```

`outputs/`（含 `joint_history_future_1024_20260920/`、
`joint_history_future_control_1024_20260920/`、`future_token_set_oracle_*_20260920/`、
各类 `run_*.sh` 队列脚本与报告）按 §9.6 不进 git。

---

## 7. 未来 token 压缩迁移：任务定义

### 7.1 研究问题

> 将 history token 的 physical compression 机制迁移到 future video token 上，
> future 能不能复用 history 的 select 参数？
>
> 如果零样本复用有希望，再通过 DriveVA 训练时相同的监督方式训练 future selector。

“相同 select 参数”建议拆成两层验证，避免混淆：

1. **选择机制参数相同**：同一 injection point、同一 source layer、同一
   `hidden_sequence`、同一 keep ratio / dynamic budget / selection operator，
   只把 domain 从 history 换成 future。
2. **selector 权重相同**：直接拿 history-trained learned selector / checkpoint，
   扩展到 future domain 做零样本推理。
3. **同监督重训**：如果零样本权重不迁移，但 oracle/冗余分析显示 future 存在可压缩空间，
   再用与 DriveVA 相同的 teacher/loss/特征层训练 future selector。

### 7.2 为什么不能直接假设会成功

- future video token 是 planning 的重要输入，不是可随意丢弃的背景 token；
- 报告中的 `Video+Action = 90.9 PDMS` vs `Action Only = 47.0` 是强烈警告：
  直接删除/大幅压缩 future video token，很可能显著掉 planning；
- history 的成功来自“模型已通过前半段网络完成语义混合”的冗余；
  future token 在同一层是否已经形成这种冗余，目前没有证据；
- 时间/跨层连续性只有略高于随机的水平，不能假设 history 的 token 排名在 future 上成立；
- 当前 learned selector 的 `_positions()` 和训练 teacher 都只支持 history。

### 7.3 当前技术支持状态

已具备：

- `FutureVideoDomain`：候选 780 个 future token；
- `ActionAttentionVNorm` / `Random` / `TokenNorm` 等通用 scorer 可对 future domain 打分；
- `AdaptiveMassSelector` 通用 dynamic budget 可对 future domain 使用；
- `hidden_sequence` + `kv_prune` 机制可以直接把 future candidate 压缩到短序列；
- synthetic future physical smoke 已通过（2026-09-17）。

尚未具备：

- 官方 runner `--domain future_video` 入口；
- `future_latent_0/1` 单独 domain；
- future 的 `each_future` budget / per-latent threshold/quota selector；
- learned selector future positions；
- 训练侧 future token capture / mask / counterfactual probe；
- future 的 oracle importance 分析与正式测试。

---

## 8. 推荐实验路线（按顺序）

### Phase F0 — 打开 future domain 的官方入口（已完成，2026-09-17）

目标：让 future 能复现 history 的基础物理压缩流程。

修改点：

1. `scripts/run_official_navsim_press.py`
   - `--domain` choices 增加 `"future_video"`；
   - 确保 `--persistent-layer-sweep`、`--persistent-mode hidden_sequence`、
     `--persistent-scorer action_attention_vnorm`、`--persistent-keep-ratio` 能在
     `future_video` 上组合；
   - 对 history-only selector（`history_threshold/history_quota`）在 future domain
     显式报错，不要静默退化。
2. `videopress/core/domain.py`
   - 增加 `future_latent_0`、`future_latent_1`（或 `video_latent_i`）domain；
   - 约定 latent 顺序：未来由近到远，或 storage index 从小到大，文档和测试必须钉死。
3. `videopress/core/budget.py` / selectors
   - 若需要双 future latent 独立配额，增加 `each_future` reference 和
     `future_threshold` / `future_quota`，或在现有的 `history_threshold` /
     `history_quota` 上泛化 domain/complete-latent 校验。
4. tests
   - 新增 `test_future_domain`；
   - 覆盖 `FutureVideoDomain` candidate 数、budget、physical hidden_sequence mapping、
     learned selector future position；
   - 更新当前显式拒绝 future 的测试
     `tests/test_online_selector.py::test_learned_selector_rejects_non_history_domains`。

Phase F0 验证命令（代码改完后）：

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework
/home/cpj/miniconda3/envs/DriveVA/bin/python -m pytest -q
```

synthetic smoke（CPU/cuda 均可，当前框架层已可跑通类似命令）：

```bash
MPLCONFIGDIR=/tmp/driveva_mpl \
/home/cpj/miniconda3/envs/DriveVA/bin/python scripts/evaluate_press.py \
  --config /tmp/future_physical_smoke.yaml \
  --output-dir /tmp/future_physical_smoke_out
```

如果 `/tmp/future_physical_smoke.yaml` 不存在，可复制
`configs/press/synthetic_physical.yaml` 后至少改成：

```yaml
press:
  name: scorer_press
  injection_point: self_attn_kv
  domain:
    name: future_video
  scorer:
    name: action_attention_vnorm
    layer: 15
    head_mode: mean
    action_mode: mean
  selector:
    name: topk
  operator:
    name: kv_prune
  budget:
    type: ratio
    value: 0.5
    reference: eligible
  cross_layer_persistence:
    enabled: true
    mode: hidden_sequence
```

官方单场景 smoke（需要先完成 `--domain future_video`）：

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework
CUDA_VISIBLE_DEVICES=0 /home/cpj/miniconda3/envs/DriveVA/bin/python \
  scripts/run_official_navsim_press.py \
  --domain future_video \
  --persistent-layer-sweep 15 \
  --persistent-mode hidden_sequence \
  --persistent-scorer action_attention_vnorm \
  --persistent-selector topk \
  --persistent-keep-ratio 0.5 \
  --max-eval-tokens 1 \
  --poc-test-derived \
  --output-root outputs/future_token_smoke
```

要求：

- 1 scene 只证明 pipeline 能跑、候选区间和 mapping 正确；
- 任何 `max-eval-tokens`/`poc-test-derived` 结果都不得写成正式结论。

### Phase F1 — Future oracle 上界与冗余分析（**当前优先级最高**）

**先不要训练 selector。** 先在 future domain 上回答：

1. future token 删除对 PDM 的敏感度有多大？
2. 是否存在一个 oracle Top-K future subset，能显著优于同预算 random？
3. attention / gradient scorer 与真正 planning attribution 的一致性有多少？

建议做法：

- 复用 history 的 counterfactual tile importance 思路，把候选范围换成 future latent；
- 在 32/64/192 scene 小面板上比较：
  - NoPress；
  - Random future top-K；
  - Attention-VNorm future top-K；
  - Gradient×Input future top-K；
  - oracle（按真实反事实 planning harm / trajectory displacement 排序的 Top-K）。
- 规模至少覆盖 64 个独立场景；最终结论必须回到 7,876 全量。

判定：

- 如果 oracle 明显优于 random，且高保留率下 PDM 近无损 → future hard selection 有希望；
- 如果 oracle ≈ random 或删除少量 future token 就大幅掉 PDM → 不要硬删 future token，
  应转向 training-time bottleneck / structured attention / merge，或放弃 future token 压缩。

F1 进展（2026-09-20）：

- tile-level oracle 已做完并失败（§4 的 2026-09-18 / 2026-09-20 条目）；
- **token-level set-level instrument 已实现**（代码 + 35 单测，见 §5/§6）：
  `videopress/oracle/`、`oracle_future_token_mask`、runner
  `--future-oracle-token-mask-jsons`、`scripts/search_future_token_set_oracle.py`；
- 队列脚本 `outputs/future_token_set_oracle_queue_20260920/run_queue.sh` 已就绪但**尚未启动**；
- 判定标准不变：只有 token-level oracle 明显优于 matched random 且高保留率近无损，
  才进入 F3 训练 future selector；否则按结论报告停止 future hard prune。

### Phase F2 — 零样本复用 history select 参数（已完成 64-scene POC，结论：零样本不迁移）

目标：回答“future 能不能直接用 history 的 select 参数”。

优先测试三组：

1. **Attention-VNorm，同 layer / 同 budget**
   - domain：`future_video`；
   - scorer：`action_attention_vnorm`；
   - layer：15、16；
   - keep ratio：0.25 / 0.5 / 0.75，以及和 history 相同的 adaptive mass 档位；
   - persistence：`hidden_sequence`；
   - 与同预算 Random control 成对比较。
2. **Learned history selector 零样本**
   - 先把 `LearnedPlanningSelectorScorer._positions()` 扩展到 future domain；
   - 建议先定义一版“storage-index”位置：
     - history latent: `t=0,1`；
     - future latent: `t=2,3`（真实 storage index）；
   - 同时保留一个兼容位置模式，以防 `t=2,3` 造成 OOD；两种模式都做 ablation；
   - checkpoint 先试：
     - `outputs/magnitude_teacher_capture_train3190_2win_20260913/step-3190.safetensors`
     - `outputs/newer_latent_selector_train3768_matched_20260912/step-3768.safetensors`
     - `outputs/older_latent_selector_train3768_20260912/step-3768.safetensors`
   - selector 先用 `topk` / 单阈值 `threshold`，再试 future 双 latent threshold；
   - 历史最佳阈值 `[0.05, 0.40]` 可以作为初始 transfer 点，但 future 顺序需要测试两种：
     - `[近, 远] = [0.05, 0.40]`（近未来保留更多）；
     - storage order 原样 `[0.05, 0.40]`（早 future / 晚 future）。
3. **Future 从零训练 selector 的前置条件**
   - 只有 F1/F2 显示 future domain 存在可学习信号时，才进入 F3。

指标：

- full 7,876 PDM、paired ΔPDM 95% CI、极端 ΔPDM/zero-score 数；
- latency mean/median、selector latency、K、hidden sequence ratio；
- 未来 token 保留率与每 latent 保留数；
- 至少 2 轮/多 seed，避免单 seed 波动。

判定：

- `CI 下界 > -0.002` 且延迟下降 ≥1%：可作为 future Press 候选；
- 若 `256` 面板好、full 7,876 掉点，按 greedy merge 教训处理：小面板不作为正式结论；
- 若零样本 history selector 不迁移但 Random/Attention 有信号，进入 F3 同监督重训；
- 若 Attention-VNorm / Random 都很快掉点，停止 future hard prune。

### Phase F3 — 用 DriveVA 相同监督训练 future selector

**Gate**：只有 F1 oracle 明显优于 matched random、且高保留率近无损时才启动；当前 64-scene 零样本 POC 不支持直接进入 F3。

目标：在 F1 证明 future 存在可压缩空间后，用与 history 相同监督范式训练 future scorer。

当前 history selector 的成功配方（参考）：

```text
model: DynamicTokenSelector(feature_mode=all)
feature_layer: 15
compression layer: 15
teacher: gradient_abs (planning loss gradient × input), teacher_keep_ratio=0.375
loss: BCE(+ optional ranking loss)
optimization: selector_only=true, lr≈3e-4, 1 epoch, 4 GPU
data: 3,190 train / 578 calibration selector capture scenes
```

future 版需要修改：

1. **训练侧 capture**
   - 在 `diffsynth/pipelines/wan_video_new.py` 增加通用 candidate range /
     `capture_selector_tokens`；
   - 保留 `capture_history_tokens` 兼容路径；
   - future 默认捕获两个 future latent，共 780 token；
   - 输出 `pipe._last_selector_tokens` / positions，或把现有
     `_last_history_tokens` 泛化并保留旧别名。
2. **训练侧 mask**
   - 把 `history_token_mask` 泛化为 candidate mask，支持 future range；
   - 旧 history 行为必须 bit-compatible。
3. **counterfactual teacher**
   - 当前 `counterfactual_latent_index` 只支持 `0..num_cond_latents-1`；
   - 增加 future candidate 表示，例如 `candidate_domain=future_video` +
     `candidate_latent_indices`；
   - `gradient_abs` teacher 可先复用，不需要新 loss；
   - 如果 gradient 信号不稳定，再试 `planning_harm` / `displacement` teacher，
     但要记录其历史结论：signed label 曾不可学习，位移 teacher 是 legacy，
     planning-harm metric-space 版本尚未形成全量证据。
4. **部署侧**
   - learned selector 支持 future positions；
   - future threshold/quota selector；
   - cross-layer persistence 使用相同的 `hidden_sequence`；
   - 测试 `source_layer`：15、16、0（pre-DiT）至少各做 smoke；
     但 history 的经验强烈提示 0 层 pre-DiT 很可能失败。
5. **评估侧**
   - 全量 7,876、2 rounds、paired bootstrap；
   - 与 NoPress、Random future、history best 做三方比较；
   - 检查 zero-score tail、`ego_progress` 新增零分、极端翻转；
   - 若目标是保留视频质量而不仅是 trajectory PDM，额外评估 video quality；
     当前 PDM 评测默认是 trajectory-only，不能代表 future video 质量。

### Phase F4 — 结论产出

至少产出以下判断：

- future token 是否存在可复用的表示冗余？
- history select 参数 / 权重是否能零样本迁移？
- 若不能，同监督重训后能否稳定优于同预算 random？
- 是否存在“加速 + 全量 PDM 近无损”的 future Press 工作点？
- 如果失败，失败在 scorer、监督目标、压缩方式，还是 future token 本身不可压缩？

---

## 9. 标准命令与工作流

### 9.1 框架测试

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework
/home/cpj/miniconda3/envs/DriveVA/bin/python -m compileall -q .
/home/cpj/miniconda3/envs/DriveVA/bin/python -m pytest -q
```

当前基线：`243 passed in 82.62s`（2026-09-20；其中
`tests/test_future_token_oracle.py` 39 个为本次 token-level oracle 新增）。

### 9.2 synthetic 功能验证

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework
MPLCONFIGDIR=/tmp/driveva_mpl \
/home/cpj/miniconda3/envs/DriveVA/bin/python scripts/evaluate_press.py \
  --config configs/press/synthetic_physical.yaml \
  --output-dir outputs/physical_round1
```

注意：synthetic backend 不是官方 PDM；只用于验证 layout/scorer/operator/evaluator 路径。

### 9.3 官方 7,876 多卡 runner（history / 通用）

参考 `videopress_framework/README.md` 的 “Official full-scene runner”：

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
/home/cpj/miniconda3/envs/DriveVA/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=6 \
  scripts/run_official_navsim_press.py \
  --metric-cache-path /mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_full \
  --score-cache-root /mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/official_press_score_cache \
  --output-root outputs/official_navsim_full \
  --force-full-scene-set \
  --enable-nuscenes-metrics
```

注意：

- full run 必须 `--force-full-scene-set`，保证 7,876 场景齐全；
- runner 对已存在 output root 会创建 `_rerunNN`，不会覆盖；
- 大规模 run 前先检查 GPU 显存、磁盘剩余和已有 output 目录。

### 9.4 当前最佳 learned history Press 复现模板

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework
CUDA_VISIBLE_DEVICES=0,1 /home/cpj/miniconda3/envs/DriveVA/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/run_official_navsim_press.py \
  --persistent-layer-sweep 15 \
  --persistent-mode hidden_sequence \
  --persistent-scorer learned_planning_selector \
  --persistent-learned-checkpoint \
    outputs/magnitude_teacher_capture_train3190_2win_20260913/step-3190.safetensors \
  --persistent-selector history_threshold \
  --per-latent-thresholds 0.05,0.40 \
  --persistent-keep-ratio 1.0 \
  --domain history \
  --rounds 1 \
  --num-inference-steps 3 \
  --force-full-scene-set \
  --output-root outputs/rerun_best_learned_history
```

如果只想 smoke，把 `--force-full-scene-set` 换成 `--max-eval-tokens 16 --poc-test-derived`。

### 9.5 结果汇总

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework
MPLCONFIGDIR=/tmp/driveva_mpl \
/home/cpj/miniconda3/envs/DriveVA/bin/python scripts/aggregate_suite.py \
  outputs/<suite_root>
```

如果输出目录里已有 `suite_manifest.json` / `suite_summary.json`，优先读原始终端日志和
`statistics/method_statistics.md`，再做结论。

---

### 9.6 仓库卫生与提交约束（用户 2026-09-17 明确要求）

- 这些提交限制必须长期生效：**不要上传一次性 bash、smoke test / 临时脚本、短期
  output、测试 log、模型/checkpoint、数据、缓存或大文件。**
- 只提交：核心源码、正式配置、稳定的单元/回归测试、设计/交接文档。
- 核心代码尽量放在 `videopress_framework/` 这个外接文件夹；`diffsynth/` 与
  `examples/` 只保留必要的 runtime hook / 训练入口集成。
- 新实验的临时 driver、GPU smoke、日志、`outputs/` 产物一律不进入 git；
  应放在 `outputs/`、`logs/` 或系统临时目录，并依赖 `.gitignore`。
- Agent 不主动 commit/push：用户要求阶段性成果后由用户发话；commit 作者固定为
  `Kasuga-future <kasuga.chen@sjtu.edu.cn>`。

---

## 10. 已知坑与注意事项

1. **绝对 PDM 协议敏感**：7,876 与 1,920 的绝对值不可直接比；只能用 paired Δ。
2. **Random control 必须等预算**：任何 hard selector 结论都要带 matched random seed 对照。
3. **小面板会骗人**：256 面板近无损的 greedy merge 在全量 7,876 上下降了约 1.08%。
4. **零分/极端场景要单独审计**：均值 PDM 会掩盖 ego_progress=0 的新增零分。
5. **延迟计算要 paired**：NoPress 和候选方法要在同批 scene/GPU/step 下比较；
   引用报告数字时写清比较基准。
6. **hidden_sequence restore 是 zeros**：Head 前 restore 会把 dropped token 置零。
   对 trajectory PDM 可能无害，对视频 decode 会产生零未来帧；不要把它当作 future video
   仍可正常解码的证据。
7. **当前 `infer_trajectory_only` 只说明 PDM**：不能代表 future video 的生成质量。
8. **learned selector 现在还排斥 future**：
   `videopress_framework/tests/test_online_selector.py::test_learned_selector_rejects_non_history_domains`
   是故意写的；迁移时必须同步改代码和测试。
9. **官方 runner `--domain` 还没有 future_video**：不要以为框架层的 `FutureVideoDomain`
   等于官方 runner 已经能跑 future。
10. **不要同时改太多变量**：future 实验要 separate layer/source/scorer/selector/persistence，
    否则无法归因。
11. **当前工作区有大量未提交改动**：先 `git diff`，不要覆盖；不要把 unrelated 修改一起提交。
12. **数据泄漏红线**：训练不能用 1,920/7,876 测试场景；训练 manifest 已做 overlap 审计，
    改动数据 split 后必须重新跑 `split_report.json` / scene overlap 检查。
13. **same-scene guard**：官方 runner 必须保留 scene boundary guard，不能把跨 scene window
    当成同一场景样本。
14. **future 顺序要钉死**：storage order、near/far 顺序、threshold 顺序必须文档化并在
    test 中固定，避免像 history temporal coordinate 那样发生静默反转。
15. **Press 的定位**：近无损加速，不保证质量提升；全量正向 ΔPDM 若不显著，不要写成提升。
16. **GPU 并发上限（2026-09-20 23:15 用户更新）**：当前**仅允许 2 张 GPU**（优先 0/1）。队列脚本的 wait_for_two_gpus 必须显式检查，且不得因他人占用而超限。
    所有 GPU 任务必须写入 tmux 队列
    `outputs/future_oracle_queue_v3_20260918/run_queue.sh`，由队列等待空闲 GPU 并执行；
    禁止直接 `nohup` 抢占 GPU。

---

## 11. Agent 快速启动清单

每次新会话建议按顺序：

1. `cd /mnt/chenpeijian/autodrive/DriveVA-lite`
2. `git status --short`；确认是否在 `main @ 7c42cee` 或新 HEAD。
3. 读本文件 §0、§7、§8。
4. 读 `videopress_framework/README.md` 中与当前任务相关的章节。
5. 如果做 future：
   - 先确认 `--domain future_video` 是否已加入官方 runner；
   - 先跑 synthetic future smoke + framework tests；
   - 再跑 1 scene official smoke；
   - 再跑 64/256 面板；
   - 最后才上 full 7,876。
6. 运行任何 official/训练实验前，先 `nvidia-smi` + `ps` 确认并发进程数 ≤ 4；
   超过 4 个必须排队（用户 2026-09-17 要求）。
7. 改代码后至少运行：
   ```bash
   cd videopress_framework
   python -m compileall -q .
   python -m pytest -q
   ```
8. 实验结束后更新 §4、§5、§6、§12，并写清：
   - 新输出目录；
   - 用的 checkpoint；
   - PDM / 延迟 / K；
   - 与 NoPress / random / history best 的 paired 比较；
   - 下一步，以及哪些结论仍不确定。

---

## 12. 会话交接日志

### 2026-09-17 — 初始化 AGENTS.md

- 创建本文件，汇总 2026-08-18 至 2026-09-17 的组会结论、代码结构、环境数据路径、
  当前最佳 Press、以及 future token 迁移的推荐路线。
- 已验证：
  - `videopress_framework` 测试当前 `184 passed`；
  - 框架层 `future_video` physical `hidden_sequence` synthetic smoke 已跑通；
  - 官方 runner 仍拒绝 `--domain future_video`，learned selector 仍拒绝 future domain；
  - 当前最佳 learned history Press 的 full run 产物已定位到
    `outputs/audit_dynamic_plan_20260913/capture_dynamic_navtest7876_seed0_rerun01/`。
- 当前 HEAD：`7c42cee`；工作区：16 modified + 4 untracked。
- 下一步（优先级从高到低）：
  1. 在官方 runner 中开放 `future_video` domain；
  2. 在 future domain 上跑 synthetic + official 1-scene smoke；
  3. 做 future oracle 上界 / random / attention-vnorm 的 64 面板对照；
  4. 如果 oracle 有信号，扩展 learned selector positions 做零样本迁移；
  5. 如果零样本不行但 oracle 可行，再按同监督训练 future selector；
  6. 任何时候得到全量结果，先看 paired CI、zero-score tail 和 latency，再写结论。

### 2026-09-17 — Future domain F0 + 64-scene 零样本迁移 POC

- 本次完成：
  - 实现 `future_latent_i` domain（storage order，near-to-far）；
  - 官方 runner 开放 `--domain future_video/future_latent_0/future_latent_1`；
  - 新增 `future_threshold` / `future_quota` 和 `each_future` budget；
  - learned selector `_positions()` 支持 `storage` / `history_compatible` future 模式；
  - 官方 persistent random control（最多 4 GPU 并发）；
  - 1-scene official smoke 通过：`outputs/future_token_smoke_20260918/`、
    `outputs/future_learned_hc_smoke_20260918/`；
  - 64-scene zero-shot POC 完成并落盘报告
    `outputs/future_poc64_report_20260917.md`。
- 测试状态：`195 passed in 81.94s`（本次 full run；此前基线 184）。
- 关键 POC 数值（truncated first 64 windows，不能当全量结论）：
  - `aa68`（K=535）ΔPDM `−0.0266`，95% CI `[−0.0710, +0.0039]`；
  - `random68` ΔPDM `−0.0331`，CI `[−0.0801, −0.0006]`；
  - `learned_hc_thr`（K=536.9）ΔPDM `−0.0651`，CI `[−0.1148, −0.0258]`；
  - `learned_hc_thr − random68 = −0.0320`，CI `[−0.0786, +0.0139]`；
  - `learned_hc_thr − learned_storage_thr = +0.0210`，CI `[−0.0005, +0.0561]`。
- 结论：future hard prune 在这套 POC 下不 near-lossless；history select 权重/阈值
  零样本不迁移，且 matched K 下不优于 random。**不要启动 full future selector 训练**，
  下一步优先做 F1 future oracle/上界；若 oracle 也无结构冗余则停止。
- 资源约束：用户明确要求最多同时占用 4 张 GPU，已写入 §3/§10/§11；本次 mid-run 曾
  误起 6 个进程，已立即终止多余 2 个，后续必须排队。
- 当前 HEAD：`7c42cee`；工作区：25 modified + 5 untracked。下一步（优先级从高到低）：
  1. 设计并跑 future counterfactual tile oracle / upper-bound（Phase F1）；
  2. 在独立 256-scene 面板复核 `aa`、`random`、`history-selector zero-shot`；
  3. 只有 F1 oracle 明显优于 matched random 且高保留率近无损，才扩展训练侧
     future capture/mask/counterfactual range 并启动同监督 future selector（F3）；
  4. 任何 full 7,876 结果都要 paired CI、extreme/zero tail、latency 一起看。

### 2026-09-17 — History-guided future compression POC

- 用户指出之前理解的偏差：future 初始是噪声，直接对 future select 近似随机；正确方向是
  用 history 的保留下位置指导 future。已按此构建：
  - `HistoryGuidedFutureSelector`（`all_video` 上先选 history，再复制 mask 到 future）；
  - `LearnedPlanningSelectorScorer(all_video_history_only=True)`（只在 history 子域打分）；
  - runner `--history-guided-future-mapping` / `--history-guided-layer` /
    `--history-guided-thresholds`。
- 已测试 mapping：`same_latent`、`reverse_latent`、`nearest_history`、`oldest_history`、
  `union_history`、`intersection_history`、`majority_history`。
- 1024-scene paired POC（4-rank，每个 rank 256 个不同场景；NoPress PDM `0.911205`）：
  - `same_latent`：K=988.9，hidden=1038.1，ΔPDM `−0.0280`，CI `[−0.0395, −0.0169]`；
  - `union_history`：K=1148.9，hidden=1197.7，ΔPDM `−0.0121`，CI `[−0.0201, −0.0043]`；
  - 同场景 history-best 参考：ΔPDM `−0.0039`，hidden=1291.9；`union_history` 额外省约
    13 ms / 94 tokens，但多掉约 0.8 PDM 点。
- 结论：history-guided prior 比直接在 noise 上选更可控，但当前阈值下 **不是
  near-lossless**；不能直接上“压缩乘 2”。下一步做 threshold / future keep-ratio
  frontier sweep，而非 future selector 训练。
- 新增报告：`outputs/history_guided_future_poc1024_report_20260917.md`。
- 测试：`199 passed`（增加 history-guided selector、runner、scorer 测试）。
- 说明：本次没有自动再 commit；用户已明确后续阶段性成果后再要求 commit。

### 2026-09-17 — history-guided frontier sweep（用户要求）

- 先按用户要求 `git` 保存进度：新增 commit `717c7fb Add history-guided future token compression`。
- 在 `history_guided_future` 上增加 `future_keep_ratio` per-future-latent cap；CLI 参数
  `--history-guided-future-keep-ratio`。
- 用同一组 common 512 paired scenes 做 sweep（NoPress PDM `0.911362`）：
  - `union_history [0.02,0.40]`：ΔPDM `−0.0004`，CI `[−0.0074,+0.0064]`，hidden 1330.8，
    latency 578.3 ms；近无损但比 history-only best 压缩更少且更慢；
  - `[0.03,0.40]`：ΔPDM `−0.0105`，hidden 1296.2；同长度下明显差于 history-only；
  - `[0.04,0.40]`：ΔPDM `−0.0093`，hidden 1249.2；
  - `[0.05,0.40]`：ΔPDM `−0.0158`，hidden 1198.1；
  - future cap 0.85/0.80/0.75/0.65：逐步变差，cap 越小 PDM 掉得越快；
  - route control：future direct `action_attention_vnorm` topk0.75 ΔPDM `−0.0276`；
    adaptive ΔPDM `−0.0094` 但 hidden 1507、latency 591.4；history-only best 在同
    512 scenes 为 ΔPDM `−0.0060`、hidden 1292.2。
- 结论：**没有找到 near-lossless 的 future 压缩乘 2 工作点**。唯一近无损的 guided 点
  压缩比 history-only 少且更慢；在相近 physical length 下 guided future 仍多掉约 0.5 PDM。
- 测试：`200 passed`；frontier 代码尚未 commit（用户要求阶段性成果后由用户发指令）。
- 新增报告：`outputs/history_guided_future_frontier_512_report_20260917.md`。
- 下一步建议：停止简单 mask-copy 调参，转向 oracle/结构化 attention / 训练期 bottleneck。

### 2026-09-17 — future subset oracle 上界初测

- 新增 `--persistent-random-seed`，用于固定/采样 matched random future mask。
- 做 best-of-N random-mask oracle（每 scene 取 N 个随机 mask 的 official PDM 最大值）：
  - 64 scenes，NoPress PDM `0.940644`；
  - future keep 0.50，N=16：single-seed ΔPDM `−0.0671`，oracle ΔPDM `+0.0026`，
    CI `[−0.0038,+0.0084]`，oracle PDM `0.9432`；
  - future keep 0.375，N=16：single `−0.1092`，oracle `−0.0209`，
    CI `[−0.0300,−0.0124]`，oracle PDM `0.9197`；
  - future keep 0.25，N=8：single `−0.1608`，oracle `−0.0709`，
    CI `[−0.0983,−0.0485]`。
- 结论：50% future keep 的 oracle 上界与 NoPress 不可区分，是 positive signal：
  存在大量 future planning 冗余子集；37.5% 仍掉 0.0209，但比随机好很多；25% 明显掉点。
  该 oracle 有选择偏差，只是 headroom upper bound，不是部署 selector。
- 下一步：实现 future counterfactual tile / planning-gradient oracle，把随机 oracle headroom
  转成可实现的 selector；再上 256/1024 面板。
- 产物：`outputs/oracle_future_subset_upper_bound_20260917.md`、
  `outputs/oracle_random_k50_seed1001..1016_20260918/` 等。

### 2026-09-18 — future counterfactual tile/mask oracle 队列

- 已实现：
  - `FutureFixedTileSelector`（`future_fixed_tiles`）：单 future latent 删除一个
    确定性 normalized tile，用于 leave-one-tile-out PDM 矩阵；
  - `OracleFutureMaskSelector`（`oracle_future_mask`）：读取 per-scene oracle tile mask JSON；
  - runner：`--future-counterfactual-tile-matrix`、`--future-counterfactual-layer`、
    `--future-counterfactual-tile-h/w`、`--future-oracle-mask-json`；
  - `scripts/build_future_oracle_masks.py`：从 matrix suite 生成
    keep-ratio 0.5/0.375/0.25 的 per-scene oracle tile mask。
- tmux 持久队列已启动：
  - session：`driveva_oracle`
  - 脚本：`outputs/future_oracle_queue_20260918/run_queue.sh`
  - 约束：**本阶段最多 2 张 GPU**；队列等待至少 2 张 GPU 各有 >=32GiB free 才启动，
    否则 sleep 60 继续等待；
  - 队列内容：2-rank x 64 = 128-scene counterfactual matrix → 三档 mask 分析 →
    oracle eval；随后 2-rank x 256 = 512-scene matrix → 三档分析 → oracle eval；
  - 当前所有物理 GPU 被 VLLM 进程占用，队列正在等待中。
- 测试：`204 passed`；matrix/oracle selector + runner builder + analysis script 均有 CPU 测试。
- 注意：本阶段新代码尚未 commit。

### 2026-09-18 — future counterfactual tile oracle 完成与结论

- tmux 队列 `driveva_oracle` 全部完成；本阶段最多 2 GPU：
  - matrix128（2×64）完成；
  - matrix512（2×256）完成；
  - 128/512 两套面板 × keep 0.50/0.375/0.25 的 oracle eval 完成。
- 512-scene oracle eval：
  - keep0.50：ΔPDM `−0.0424`，95% CI `[−0.0637,−0.0220]`；
  - keep0.375：ΔPDM `−0.0891`，CI `[−0.1129,−0.0665]`；
  - keep0.25：ΔPDM `−0.1129`，CI `[−0.1385,−0.0882]`。
- 128-scene oracle eval 同样差：keep0.50 `−0.0610`，keep0.375 `−0.1080`，
  keep0.25 `−0.1082`。
- matrix512 tile-level 统计：mean harm `[-0.004,+0.009]`，SE `0.003–0.006`，
  `harm>0` 比例 `0.23–0.45`；单次 PDM tile 排序基本被噪声/交互主导。
- 结论：
  - 当前 counterfactual tile oracle 不是有效 oracle，明显差于 history-guided 和
    attention-vnorm controls；
  - tile 重要性不满足可加性，不能用“单 tile 删除 PDM”外推组合子集；
  - 该结果不证明 future 不可压缩，只证明该 oracle 构造失败；
  - 下一步改 trajectory displacement / multi-replay / set-level greedy/beam oracle，
    并补 matched random control。
- 产物：`outputs/future_oracle_counterfactual_conclusion_20260918.md`。

### 2026-09-20 — multi-seed trajectory/planning-harm oracle 队列启动

- 根据用户要求，不再只用单次 PDM：
  - runner 增加 `--dump-target-trajectories`，由官方 target builder 自动保存 GT trajectory；
  - 已有 `--dump-trajectories` 保存预测 trajectory；
  - 新增 `scripts/build_future_oracle_masks_multiseed.py`，可对多个 seed 的
    `pdm_harm` / `traj_disp` / `planning_harm` 聚合，按 `traj_disp`、
    `planning_harm`、`combined` 等 ranking 生成 oracle mask。
- 新 tmux 队列 `driveva_oracle_v2`：
  - 脚本：`outputs/future_oracle_queue_v2_20260918/run_queue.sh`
  - 4 个 sample seed：1001/1002/1003/1004；
  - 每个 seed 跑 2-rank × 64 = 128-scene counterfactual tile matrix，同时 dump
    predicted + target trajectories；
  - 然后对 3 种 ranking（`traj_disp`、`planning_harm`、`combined`）× 3 个 keep ratio
    （0.50/0.375/0.25）生成 mask 并跑官方 oracle eval；
  - 本阶段严格最多 2 张 GPU。
- 队列启动时间：`2026-09-20 10:08:58`，当前 seed1001 matrix 正在运行。
- 说明：这一版的目标是确认 oracle upper bound，而不是优化 oracle 分数；如果发现对正常
  pipeline 有效的选择 trick，则优先接入正常 selector。

### 2026-09-20 — 切换到 4 GPU / multi-seed trajectory oracle

- 用户更新资源约束：自此可同时使用 **4 张 GPU**；已同步 §3.1、§10、§11。
- 旧 v2 2-GPU 队列已停止；新队列：
  - tmux：`driveva_oracle_v3`
  - 脚本：`outputs/future_oracle_queue_v3_20260918/run_queue.sh`
  - 4 ranks × `--max-eval-tokens 64` = 每个 seed 约 256 个独立 scene；
  - 4 个 sample seed：1001/1002/1003/1004；
  - 同时 dump predicted trajectory + target trajectory；
  - 使用 `scripts/build_future_oracle_masks_multiseed.py` 聚合
    `pdm_harm` / `traj_disp` / `planning_harm`，生成
    `traj_disp`、`planning_harm`、`combined` 三类 mask；
  - 最后对 3 ranking × 3 keep ratio = 9 个 oracle eval。
- 当前 `matrix_seed1001` 正在 GPU 0/2/3/4 上运行。
- 目的：确认 oracle 上界，寻找对正常 pipeline 有效的 selection trick，而不是优化 oracle
  分数本身。

### 2026-09-20 — 4-GPU multi-seed trajectory/planning oracle 完成

- `driveva_oracle_v3` 全部完成，最多 4 GPU：
  - 4 sample seeds × 256 scenes；
  - 每个 tile 同时有 PDM、predicted trajectory、target trajectory；
  - 3 ranking（traj_disp / planning_harm / combined）× 3 keep ratio。
- 最终 oracle eval：
  - 最好的是 `combined keep=0.50`：ΔPDM `−0.0182`，CI `[−0.0445,+0.0074]`；
  - `traj_disp keep=0.50`：ΔPDM `−0.0453`；
  - `planning_harm keep=0.50`：ΔPDM `−0.0473`；
  - keep 越低越差。
- 结论：
  - tile-level oracle 仍不 near-lossless，且不如 history-guided / attention-vnorm controls；
  - trajectory displacement / planning harm 比单次 PDM 稳定，但没有转化成有效 token 重要性标签；
  - 更根本的问题可能是 **tile 粒度**：之前 64-scene token-level best-of-N random-mask oracle
    在 50% keep 接近 NoPress，而 tile oracle 只能选 12 个 tile 子集；
  - 因此这是 tile-constrained oracle 失败，不是 future token 不可压缩的证明。
- 下一步：token-level set-level oracle（best-of-N random token masks、greedy/beam），
  继续同时测 PDM + trajectory displacement + planning harm。
- 产物：`outputs/future_oracle_multiseed4_conclusion_20260920.md`。

### 2026-09-20 — token-level set-level future oracle 实现（代码 + 测试完成，实验未跑）

- 用户要求把 future token oracle 从 tile 级细化到 **token-level set-level**，并按
  `outputs/future_oracle_multiseed4_conclusion_20260920.md` 的“下一步建议”完成代码与测试。
- 本次新增（尚未 commit，均为工作区内改动）：
  - `videopress/oracle/token_set.py`：token 分组（`token`/`linear`/`block`/`random`，绝不跨
    latent）、flat offset ↔ per-latent JSON 互转、`SetScorer`（联合打分 + 缓存 + 批量）、
    `greedy_forward_selection` / `greedy_backward_elimination` / `beam_search` /
    `random_search`（matched-budget random）、`random_token_masks`（per-scene best-of-N）、
    `per_scene_oracle`。
  - `videopress/oracle/metrics.py`：`trajectory_displacement` / `planning_harm` /
    `combined_harm`（z-score）/ `OBJECTIVES` / `objective_higher_is_better`。
  - `videopress/selectors/future_oracle.py`：新增 `oracle_future_token_mask`（per-scene
    per-latent token 索引或 flat offset，越界/缺场景显式报错），并把 tile/token 两个 oracle
    selector 的公共物理应用逻辑抽成 `_FutureMaskSelector`；tile oracle 行为保持不变
    （原测试继续通过）。
  - runner：`--future-oracle-token-mask-json` / `--future-oracle-token-mask-jsons`
    （后者每个 mask 文件一个 physical method，整批候选一次 runner 调用）；
    `_future_oracle_token_mask_paths` / `_mask_method_tag` / `_future_token_mask_specs`。
  - `scripts/search_future_token_set_oracle.py`：模式 `random-best-of-n` /
    `independent-topk` / `greedy-forward` / `greedy-backward` / `beam` / `evaluate-masks`；
    每个候选同时记录 PDM / pdm_harm / traj_disp / planning_harm；自适应模式附 matched
    random control；所有 runner 命令写入 `search/commands.jsonl`；容忍 truncated POC 的
    runner exit 1（artifacts 完整时继续）；支持 `--dry-run`。
  - `tests/test_future_token_oracle.py`：36 个 CPU 单测（分组、搜索、序列化、per-scene
    oracle、metric、selector、runner spec、driver 命令/dry-run 辅助）；其中
    `test_beam_search_spends_the_budget_on_score_ties` 是队列启动后发现的回归测试：
    原 `beam_search` 在候选分数完全打平时会停在更小的子集（甚至空集），等于用
    “少压缩”伪装成 oracle 更优；已改为同分时优先选 token 更多的状态。
  - `outputs/future_token_set_oracle_queue_20260920/run_queue.sh`：4-GPU 队列脚本
    （random best-of-N / independent top-K / greedy forward / beam），**已就绪但未启动**。
  - `videopress_framework/README.md`：新增 “Token-level set-level future oracle” 章节。
- 测试与验证：
  - `python -m pytest -q`：**240 passed**（新增文件 35 个测试全通过；此前 184/195/199/204
    的历史计数见 §9/§12）；
  - `python -m compileall` 通过；
  - driver `--dry-run` 可生成 mask JSON 并打印完整 runner 命令；
  - **GPU 1-scene official smoke 通过**（`--max-eval-tokens 1`，2 个随机 token mask，
    keep 0.5）：
    - 产物 `outputs/future_token_set_oracle_smoke_20260920/`；
    - baseline（NoPress）PDM `0.928989`（单场景，sample seed 默认）；
    - `sample000` PDM `0.482048`，`sample001` PDM `0.943476`，per-scene best-of-2 oracle
      `0.943476`；
    - 物理路径确认：`K=390/780`，`hidden_sequence_length=1179`（1569−390），
      selector=`oracle_future_token_mask`；
    - 该 smoke 只证明 token mask 能真正走物理 `kv_prune`+`hidden_sequence` 且 driver 端到端
      可跑，**不是 PDM 结论**（单场景 + truncated）。
- 注意：**本次没有产生新的 PDM 结论**。tile oracle 的失败结论仍然有效；token-level
  set-level 只是把 instrument 做对，需要跑队列才有结论。
- 下一步：
  1. 启动 `outputs/future_token_set_oracle_queue_20260920/run_queue.sh`（保持 ≤4 GPU）；
  2. 先看 `random-best-of-n` 的 per-scene oracle 能否在 256-scene 面板复现“50% keep
     接近 NoPress”；再看 greedy/beam 是否显著优于 matched random；
  3. 若 token-level set-level oracle 也失败，按结论报告停止 future hard prune，
     转向 structured attention / training-time bottleneck。

### 2026-09-20 — token-level oracle 队列已启动 + ETA

- tmux `token_oracle` 启动 `outputs/future_token_set_oracle_queue_20260920/run_queue.sh`
  （13:19:40 起，最多 4 GPU）：`random_best_of_n`（256 场景）→ `independent_topk_g8`
  → `greedy_forward_g30` → `beam_w2_g30`（后三档用 64 场景 panel）。
- 实测吞吐（4 ranks、每 method 256 场景）：**63 s/method**；v3 matrix 校准一致
  （25 methods / 26m35s）。
- ETA（按各阶段 method 数与 invocation 数估算，含每次 runner 调用的约 45 s 固定开销）：
  - stage1 `random_best_of_n`：17 methods ≈ 19 min → **约 13:38 出报告**；
  - stage2 `independent_topk_g8`：101 methods（含 compose）≈ 1h48m → 约 15:26；
  - stage3 `greedy_forward_g30`：294 methods / 16 invocations ≈ 5h21m → 约 20:47；
  - stage4 `beam_w2_g30`：517 methods / 16 invocations ≈ 9h15m → 约次日 06:02；
  - 合计约 **16h42m**，预计 **2026-09-21 06:00 左右**全部完成。
- 备注：每个 candidate runner 调用都会附带一次 in-suite `physical_no_press`（driver 已有
  外部 baseline），全部阶段合计约 29 个冗余 method ≈ 30 min；若需要可加
  `--persistent-skip-baseline` 收紧，但需在阶段边界改代码。
- 测试：beam tie 回归修复后 full run **240 passed**。

### 2026-09-20 — stage1 结果 + `--persistent-skip-baseline`

- stage1 `random_best_of_n` 13:38:33 完成 rc=0（256 场景，基线 NoPress PDM `0.880333`）：
  - 单个随机 token mask（keep 0.5）panel 均值 PDM `0.866133`，ΔPDM `−0.0142`；
  - 16 个 mask 的 per-scene best-of-16 oracle `0.939807`，ΔPDM `+0.059474`，
    paired bootstrap 95% CI `[+0.0331, +0.0889]`（不跨 0）；
  - 已用原始 records 独立复核：oracle 与报告**逐场景完全一致**；
  - **重要 caveat**：144/256（56%）场景没有任何一个 mask 能超过 NoPress；per-scene
    跨 16 个 mask 的 PDM 极差均值 `0.200`、逐场景 sd 均值 `0.068`，而 16 个 arm 的
    panel 均值只散布在 `0.8513–0.8825`。即 per-scene max 主要是在大噪声上取最大值，
    选择偏差很重，`+0.059` **不能读成"存在可部署的近无损子集"**。可部署的对照应是
    matched random 的 panel 均值 `0.866`，看 greedy/beam 能否超过它。
- 代码：driver 新增 `--keep-suite-baseline`；默认给 candidate runner 调用加
  `--persistent-skip-baseline`（driver 已有外部 baseline）。stage2 的 driver 进程在
  改动前已启动，因此 stage2 不受影响；stage3/4 生效，各节省 16 个冗余 method
  （约 17 min/阶段）。测试 `240 passed`；dry-run 已确认命令带该 flag。
- 更新后的 ETA：stage2 ≈ 15:30；stage3（5h04m）≈ 20:35；stage4（8h58m）≈ 次日 05:30。

### 2026-09-20 — stage2 结果、64-scene 噪声地板与“是否需要继续测试”的决策

- stage2 `independent_topk_g8`（15:27:19 rc=0，64 场景，基线 PDM `0.872972`）：
  - leave-one-8-token-group-out（保留 772/780）98 个组：**全部 dPDM 为正**，
    均值 `+0.0279`，组间 sd `0.0075`，min `+0.0029`，max `+0.0441`；
  - independent top-K 组合（保留 390）：PDM `0.844794`，dPDM `−0.0282`，
    paired 95% CI `[−0.0925,+0.0330]`，improved/tied/worse `20/20/24`。
  - 结论：token-group 粒度上“独立排序再组合”**再次失败**（与 tile oracle 同结论）；
    而且“丢掉 1% token 反而 +2.8 分”直接暴露了面板噪声量级。
- **64-scene 面板噪声地板（用 stage1 的 16 个随机 mask 在同一 64 场景上测）**：
  - 单个随机 mask dPDM 范围 `−0.0685…−0.0075`，均值 `−0.0343`，sd `0.0197`；
  - 固定 mask vs NoPress 的 paired 95% CI 宽度 ≈ `0.057`（256 场景 ≈ `0.051`，
    1024 场景外推 ≈ `0.026`）；
  - 随机 mask 两两之间 |dPDM| 均值 `0.0098`、最大 `0.0313`；
  - **1 个场景翻转 = 0.0156 PDM（64 场景）/ 0.0039（256）/ 0.0010（1024）**。
  - 即 stage3/stage4 在 64 场景上的关键比较分辨率只有 ±0.025–0.05，
    而我们要找的是 <1 分的效应 → **stage3 结构上无法给出结论，stage4 预期信息增益更低**。
- 决策建议（待用户确认）：**取消 stage4（beam，约 9h）**，保留 stage3 以产出候选 mask，
  然后把 stage3 的 `search_final`（+ stage2 的 composed 作为对照）放到
  **1024 场景（4 个 v3 seed 面板）**上做配对验证（约 15–25 min），
  判定规则预先固定：必须优于同面板 best-of-16 随机 mask 且 paired ΔPDM 的 CI 下界 > `−0.002`，
  否则停止 future hard prune，转 structured attention / training-time bottleneck。
- driver 已补上决策级统计：每个候选输出 `vs_baseline`（paired ΔPDM、95% CI、
  improved/tied/worse、extreme flips、zero cand/base），自适应模式另外输出
  `final_vs_matched_random` paired CI；测试 `241 passed`。

### 2026-09-20 — 采样种子噪声标定 + 取消 beam、改为 1024 场景判决性验证

- 用户决定：**取消 stage4（beam）**，保留 stage3，随后跑 1024 场景验证。
- **采样种子噪声标定（用 v3 matrix：同一个 24 个 tile mask × 4 个 `--sample-seed`，
  同样 256 场景）**——这是本项目一个重要方法学结论：
  - NoPress 的**绝对** panel 均值随采样种子变化 `0.880333 / 0.904601 / 0.911602 / 0.901051`，
    极差 `0.0313`；**绝对 PDM 不能跨采样种子比较**；
  - 但同一个固定 mask 的**配对** ΔPDM 跨种子只有 sd `0.0067`、极差 `0.0171`；
  - 另一个标定：随机 mask 两两之间 |ΔPDM| 均值 `0.0098`、最大 `0.0313`；
    固定 mask vs NoPress 的 paired 95% CI 宽度 ≈ `0.057`（64 场景）/ `0.051`（256）
    / 外推 `0.026`（1024）。
  - 结论：**配对比较是唯一有效货币**；提升功效主要靠**增加场景数**（而不是加种子），
    因为 256 场景上总噪声 sd≈0.013，其中种子分量≈0.007、场景×mask 交互≈0.011。
- 因此 stage3/4 在 64 场景上的关键比较分辨率只有 ±0.025–0.05，**结构上无法判定 <1 分的效应**。
- 已实现（未 commit）：
  - 守卫脚本 `outputs/future_token_set_oracle_queue_20260920/stop_after_stage3.sh`
    （tmux `oracle_guard`）：stage3 一结束就杀掉队列、阻止 beam，并自动在
    tmux `token_verify` 里启动验证；
  - 验证脚本 `outputs/future_token_set_oracle_verify_1024_20260920/run_verify.sh`：
    把 stage3 的 `search_final` 与 stage2 的 `topk_composed` 折叠成 `{"*": pattern}`
    广播 mask，在 **4 rank × `--max-eval-tokens 256` = 1024 场景**上评估，
    同时跑 15 个同预算随机 mask 作为 matched random band（约 76 min）；
  - selector 支持 `"*"` 广播条目（共享 pattern 不再需要逐场景展开）；
  - driver `--mode evaluate-masks --include-random-masks N`：同一次 runner 调用里
    同时评估命名 mask 与随机 band。
- 预先固定的判定规则：搜索 mask 必须 (a) 超过随机 band，(b) 相对 NoPress 的 paired
  ΔPDM 的 CI 下界 > `−0.002`；否则**停止 future hard prune**，转 structured attention /
  training-time bottleneck。
- 测试：新增 2 个（`"*"` 广播语义、`build_evaluate_mask_candidates` 随机 band）。

### 2026-09-20 — 1024 场景验证完成：结论与下一步判断

- 全部完成：stage1 `random_best_of_n`（13:38）、stage2 `independent_topk_g8`（15:27）、
  stage3 `greedy_forward_g30`（15:27→19:25）、**stage4 beam 被看门狗按计划阻止**、
  1024 场景验证（19:25→20:35）。
- 1024 场景 keep 0.5 结果（基线 `0.911205`）：
  - 随机 band（15 臂，K=390）ΔPDM `−0.0557…−0.0413`（mean `−0.0483`，sd `0.0036`）；
  - stage2 composed（K=390）ΔPDM `−0.0534` CI `[−0.0686,−0.0386]`，只胜过 13% 随机臂；
  - stage3 greedy（K=270，预算不匹配）ΔPDM `−0.0726`，输给全部 15 个随机臂；
  - zero-score：候选 80/81 vs 基线 30；extreme 71/73。
- **判断**：
  1. F1 gate **否定关闭**——没有 token-level oracle 子集在 50% keep 上优于 matched random；
     **取消 F3（训练 future selector）**，没有可蒸馏信号；
  2. 已推翻"50% future keep 存在大量冗余"的旧乐观结论（64/256 场景 POC 的产物）；
  3. 唯一还值得测的是 **keep-ratio frontier**（不是更多搜索）：已在
     `outputs/future_token_set_oracle_followup_20260920/` 启动（270-token 预算匹配随机 band
     + keep 0.75/0.875 的 1024 场景随机控制，约 50 min）；
  4. 若 keep 0.75 仍远非近无损 → 彻底放弃 future hard prune，保留已部署的 history press，
     转 structured attention / training-time bottleneck / merge / quantization；
  5. 方法学教训写入 §10：小面板搜索会过拟合，禁止用 64 场景搜索结果作为可部署 pattern。
- 测试：`243 passed`。

### 2026-09-20 — 补测完成：keep-ratio frontier 与预算匹配修正（本阶段定论）

- follow-up（21:24→22:09，1024 场景、NoPress `0.911205`）结果：
  - keep 0.346 / 0.500 / 0.750 / 0.875 的随机控制 dPDM 分别为 `−0.0978` / `−0.0483` /
    `−0.0216` / `−0.0096`；**所有 keep ≤ 0.875 的臂 CI 均不跨 0（显著差于 NoPress）**；
  - **预算匹配修正**：searched 270-token mask `−0.0726` vs 270-token 随机 band `−0.0978`，
    searched 胜 `+0.0187` CI `[+0.0029,+0.0339]` → selection 匹配预算下**确实有信号**，
    但绝对水平不可用；此前"不优于随机"的说法是预算不匹配造成的假象，已更正；
  - zero-score：30 → 37–45(.875) → 51–55(.75) → 65–77(.346)。
- **本阶段结论**：token-level set-level oracle 无法让 future hard prune 近无损；keep≥0.875 的
  代价（−0.007~−0.015）换不到有意义的加速（仅 6.2% 序列长度）。→ **停止 future hard prune，
  取消 F3**；保留 history press；能力侧转 structured attention / training-time bottleneck。
- 未解决项：只测了 trajectory PDM；video decode 质量必然因 hidden_sequence 置零而受损；
  keep 0.875 上的可部署 scorer 未测（可选 5 min 补测）。

### 2026-09-20 — 收尾：future hard prune 关闭，仓库上传，转入 history+future 联合搜索

- **收尾检查**（keep 0.875，1024 场景）：可部署 `action_attention_vnorm` ΔPDM `−0.0133`
  CI `[−0.0211,−0.0058]`，只胜过 33% 随机臂，paired vs best random `−0.0065`
  `[−0.0155,+0.0024]`，zero 44/30 → **唯一近中性档位也不可用**；
- 结论报告：`outputs/future_token_level_oracle_conclusion_20260920.md`
  （含全部结果、被推翻/更正的旧结论、可复用方法学、原始输出索引）；
- **仓库上传**：commit `87376e8`（token-level oracle 代码 + 39 单测）与
  `9549593`（文档/交接）已 push 到 `origin/main`
  (`https://github.com/Kasuga-future/DriveVA-lite.git`)，author/committer 均为
  `Kasuga-future <kasuga.chen@sjtu.edu.cn>`（按 §9.6 约定）；
  `outputs/` 产物与 `scripts/pre_dit_gpu_smoke.py`（smoke/临时脚本）按 hygiene 未入库；
- **下一步（已启动）**：history+future 联合最佳点搜索
  `outputs/joint_history_future_1024_20260920/run_joint.sh`（1024 场景，
  以部署的 history-only press 为参照，测 `union_history` / `same_latent` 及 future cap
  0.875/0.75 的联合 PDM / 延迟 / K / 零分尾部；tmux `joint_search`，约 40 min）。

### 2026-09-20 — history+future 联合搜索完成（1024 场景）+ 资源收紧到 2 GPU

- **联合搜索 7/7 臂完成**（23:05:22，1024 场景，NoPress `0.911205`，全部同一面板）：

| arm | K | hidden | lat ms | ΔPDM | 95% CI | zero |
|---|---:|---:|---:|---:|---|---:|
| **history_only（部署参照）** | 490 | 1279 | 575.9 | **−0.0039** | [−0.0081,−0.0004] | 34/30 |
| union_history | 1149 | 1158 | 571.1 | −0.0121 | [−0.0203,−0.0046] | 44/30 |
| union_history_cap0875 | 1133 | 1142 | 578.6 | −0.0160 | [−0.0257,−0.0069] | 49/30 |
| same_latent | 989 | 998 | 556.0 | −0.0280 | [−0.0395,−0.0167] | 49/30 |
| union_history_cap075 | 1072 | 1081 | 570.1 | −0.0280 | [−0.0392,−0.0174] | 60/30 |
| same_latent_cap0875 | 982 | 991 | 570.1 | −0.0285 | [−0.0403,−0.0169] | 50/30 |
| same_latent_cap075 | 955 | 964 | 545.7 | −0.0299 | [−0.0421,−0.0182] | 54/30 |

- **结论**：联合前沿**单调、无拐点**——在 history-only 之上每多砍 ~120–150 hidden token
  约付 1 个 PDM 点；延迟不与 hidden 单调（union 571 ms vs history-only 576 ms，几乎无收益，
  cap0.875 反而 578.6 ms），只有 `same_latent*` 有 6–30 ms 明确加速；**没有任何联合臂达到
  近无损门槛**（CI 下界 > −0.002，最好 union_history 为 −0.0203）。**联合最佳可部署点仍是
  history-only**。零分尾部随压缩单调恶化：30 → 34 → 44 → 49 → 54–60。
- **matched-K 随机对照**：原控制脚本用了 `--domain all_video`，而 runner 的 `--domain`
  choices 里没有它 → 4 臂 argparse 立即失败（已修：runner 新增 `all_video` choice + 单测；
  测试 `244 passed`）。重启后 4 臂由 `run_joint_control_2gpu.sh` 在
  `outputs/joint_history_future_control_1024_20260920/` 下运行。
- **只占用真正空闲的 GPU（用户 23:2x 要求）**：`acquire_gpus()` 只接受 free ≥ 40 GiB 的卡；
  有两张空闲就用两张（nproc=2），一张空闲都没有就等待，只有一张长期空闲（>15 min）才降级到
  nproc=1，绝不与他人共卡。首次启动选中的 0+2 是错的：GPU 2 上 zhanglizhong 的 dino_dit
  评测 3 分钟内从 24 GiB 涨到 48 GiB；已终止并改到 **0+3**（两张均只有 3 MiB 占用）。
- **面板口径**：官方 evaluator 按 rank 分块（rank r 取第 r 块再截断 `--max-eval-tokens`），
  所以 world_size=2/max=256 覆盖 `[0,256)+[N/2,N/2+256)` = 512 场景，是 1024 联合面板的
  **子集**；分析脚本自动取共同子集做配对比较。
- **资源收紧**：用户 23:15 要求**只能用 2 张 GPU**，23:2x 进一步要求**只占用空闲卡**；
  AGENTS §3.1/§10 已同步，队列改用 `acquire_gpus()`（free ≥ 40 GiB 才算空闲）。实测 GPU 1
  的 22 GiB 属 zhanglizhong 的 dino_dit 任务（非本会话残留），故控制最终跑在 **0+3**。
- **进程审计**：GPU 上无本会话孤儿；3228545=zhanglizhong dino_dit、3235490/91=xiangyike
  hidden_gradient_ablation、3191529/30=VLLM。注意：本会话曾用 `pkill -f run_official_navsim_press`
  误匹配到 xiangyike 的同名脚本（TERM 未生效，任务存活），后续必须按 PID/仓库路径确认归属。

### 2026-09-20 — matched-K 全 video 随机对照完成：联合 selector 有真实信号，但绝对损失仍不可部署

4 臂控制 23:20:50→23:40:09 完成（`outputs/joint_history_future_control_1024_20260920/`，
`--domain all_video --persistent-scorer random --persistent-selector topk`，nproc=2/max=256 →
**共同 512 场景子集**，NoPress `0.911362`）。预算匹配已逐臂校验：
ratio 0.7365 → `n_kept=1149`/`hidden=1158`（正好等于 `union_history`），
ratio 0.6340 → `n_kept=989`/`hidden=998`（正好等于 `same_latent`）。

| arm | K | hidden | ΔPDM vs NoPress | 95% CI | lat ms | zero cand/base |
|---|---:|---:|---:|---|---:|---:|
| `union_history` | 1149 | 1158 | **−0.0158** | [−0.0281,−0.0051] | 573.7 | 25/15 |
| `same_latent` | 989 | 998 | −0.0321 | [−0.0486,−0.0165] | 557.8 | 27/15 |
| `history_only` | 490 | 1279 | −0.0060 | [−0.0130,−0.0005] | 582.5 | 18/15 |
| `random_k1149_seed0` | 1149 | 1158 | −0.0386 | [−0.0557,−0.0228] | 583.7 | 35/15 |
| `random_k1149_seed1` | 1149 | 1158 | −0.0396 | [−0.0585,−0.0220] | 580.4 | 36/15 |
| `random_k989_seed0` | 989 | 998 | −0.0332 | [−0.0516,−0.0156] | 525.7 | 30/15 |
| `random_k989_seed1` | 989 | 998 | −0.0501 | [−0.0699,−0.0317] | 532.5 | 40/15 |

配对（同 512 场景，bootstrap 20000，seed 20260920）：

- `union_history − random_k1149_seed0` = **`+0.0228`** CI `[+0.0081,+0.0387]`，**不含 0**
- `union_history − random_k1149_seed1` = **`+0.0238`** CI `[+0.0062,+0.0420]`，**不含 0**
  → **`union_history` 胜过 2/2 matched-K 随机臂**（随机均值 `−0.0391`）
- `same_latent − random_k989_seed0` = `+0.0011` CI `[−0.0174,+0.0199]`，跨 0
- `same_latent − random_k989_seed1` = `+0.0181` CI `[−0.0023,+0.0393]`，跨 0
  → `same_latent` 名义上 2/2 更好，但**优势不显著**（两个随机 seed 自身差 0.0169，
  该预算下随机 band 噪声更大）

**结论（verified）**：

1. **history-guided 的 selection 信号是真实的**：`union_history` 在完全相同的 K=1149 /
   hidden=1158 上比随机全 video 剪枝高 `+0.023` PDM，两个 seed 的 CI 都不跨 0。
   这是本项目少见的“selector 显著优于 matched random”的正向证据（此前多为不显著或更差）。
2. **但绝对损失仍是门槛的 8 倍**：`union_history` 自身 ΔPDM `−0.0158`，离 near-lossless
   `−0.002` 还差一个数量级；`same_latent` `−0.0321` 更差。→ **联合 press 依然不可部署**。
3. **`same_latent` 丢掉了信号**：把 history mask 原样复制到同一 latent 的 future，虽然
   压缩更多，但相对随机的优势不再显著（`+0.001`/`+0.018`）。用户“同位置复制 ⇒ 乘 2”
   的直觉在 union 形态下成立、在 same-latent 形态下不成立。
4. **面板交叉校验通过**：512 子集与 1024 面板同向（union `−0.0158` vs `−0.0121`；
   same_latent `−0.0321` vs `−0.0280`；history_only `−0.0060` vs `−0.0039`），
   512 子集略悲观但排序一致。
5. **延迟有未解释的结构性差异**：`same_latent` 比同长度随机慢 `+25…+32 ms`
   （CI 不含 0），而 `union_history` 比同长度随机快 `−7…−10 ms`。两者 hidden 长度相同，
   说明**延迟不只由隐藏序列长度决定**，可能与 kept 索引的分布/分配器状态有关；
   引用时只能写“同批实测 + CI”，不要归因机制。
6. **零分尾部**：控制臂 30–40，联合臂 18–27（基线 15）——联合臂的零分尾部反而比
   同等预算随机更小，与 selection 有信号一致，但仍高于 NoPress。

**下一步（本轮结束）**：future hard prune 与联合 press 两条线都已在 1024 场景上关闭；
联合最佳可部署点仍是已部署的 history-only press。若要继续，方向是
structured attention / training-time bottleneck / merge / quantization，而**不是**
继续搜索 token 子集。产物：`/tmp/joint_control_analysis.py`（分析脚本，未入库）
与上述 `outputs/` 目录。
