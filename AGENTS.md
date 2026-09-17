# AGENTS.md — DriveVA-lite Video Token Compression 交接文件

> 最后更新：2026-09-17 19:35 CST
> 当前分支：`main`，当前 HEAD：`7c42cee`（Develop dynamic VideoPress selection and validation）
> 当前工作区：25 个 modified + 5 个 untracked，另加本文件（见 §6）
> 当前主任务：Future token compression 的 Phase F0 已打通（官方 runner `future_video` /
> `future_latent_i`、future 阈值/quota selector、learned selector future positions、
> random persistent scorer）。Phase F2 零样本迁移做了 64-scene POC：future 硬删在
> 相近保留率下显著掉 PDM，history-trained learned selector 未优于 matched random。
> 当前结论是 **不要立即开始 future selector 全量训练**，先补 Phase F1 的 future
> oracle/上界分析；若 oracle 也没有结构冗余，则停止 hard future prune。
> **资源约束（用户 2026-09-17 明确要求）**：任何实验/agent 任务最多同时占用
> **4 张 GPU**，超过 4 个进程必须排队。
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
- **资源约束**：最多同时占用 4 张 GPU；本文件 §10 已同步。

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
- **并发上限（2026-09-17 用户要求）**：最多同时占用 4 张 GPU，最多 4 个模型/训练/
  评测进程；多余任务排队，启动前用 `nvidia-smi` 和 `ps` 检查。
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
| future domain 官方 runner | **已开放** | `--domain future_video/future_latent_0/future_latent_1` 可用 |
| future latent 单独 domain/budget | **已实现** | `future_latent_i`、`each_future` reference 已加入并有测试 |
| future online selector 训练 | **未实现** | `capture_history_tokens` / `history_token_mask` / `counterfactual_latent_index` 仍只覆盖 history |
| future oracle/上界分析 | **未实现** | `counterfactual_latent_index` 只覆盖 history；这是 F3 训练前的 gate |
| future physical smoke | 已跑通 official single scene + 64-scene POC | `outputs/future_token_smoke_20260918/`、`outputs/future_poc64_report_20260917.md` |

---

## 6. 当前工作区未提交状态（2026-09-17）

`git status`：`main @ 7c42cee`，25 个 modified + 5 个 untracked。当前工作区仍包含历史实验改动与本次 future-domain 改动，切换/提交前必须逐 diff review。

### 6.1 Modified（需要 review / commit）

```text
M diffsynth/pipelines/wan_video_new.py
M diffsynth/trainers/utils.py
M examples/wanvideo/driveva_train/scripts/train_navsim_v1.sh
M examples/wanvideo/driveva_train/train_navsim_v1.py
M videopress_framework/README.md
M videopress_framework/scripts/run_full_compression_suite.py
M videopress_framework/scripts/run_official_navsim_press.py
M videopress_framework/tests/test_framework_repairs.py
M videopress_framework/tests/test_online_selector.py
M videopress_framework/tests/test_videopress.py
M videopress_framework/videopress/__init__.py
M videopress_framework/videopress/adapters/driveva.py
M videopress_framework/videopress/core/budget.py
M videopress_framework/videopress/core/domain.py
M videopress_framework/videopress/core/plan.py
M videopress_framework/videopress/factory.py
M videopress_framework/videopress/operators/__init__.py
M videopress_framework/videopress/operators/merge.py
M videopress_framework/videopress/presses/__init__.py
M videopress_framework/videopress/presses/merge_press.py
M videopress_framework/videopress/presses/scorer_press.py
M videopress_framework/videopress/scorers/learned_selector.py
M videopress_framework/videopress/selectors/__init__.py
M videopress_framework/videopress/selectors/history_budget.py
M videopress_framework/videopress/training/online_selector.py
```

### 6.2 Untracked（新文件）

```text
?? AGENTS.md
?? videopress_framework/scripts/pre_dit_gpu_smoke.py
?? videopress_framework/tests/test_pre_dit_selection.py
?? videopress_framework/videopress/operators/hidden_prune.py
?? videopress_framework/videopress/presses/learnable_merge.py
```
这些改动主要覆盖：

- `BLOCK_INPUT` / pre-DiT hidden pruning；
- hidden-sequence cross-layer controller；
- pre-DiT merge / learnable merge / register bottleneck；
- online selector 的 `pre_dit` counterfactual injection、planning-harm 指标透传；
- `2026-09-17` 的 learned history + `history_threshold` 最佳配置；
- 本次 future 迁移：future latent domain / budget / threshold/quota selector、
  runner future CLI、learned selector future positions、persistent random control。

**注意**：当前工作区不是 clean state。新 agent 在切换任务/提交前先看清楚 diff，
不要覆盖别人的未提交工作；运行测试至少覆盖本次修改。

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

当前基线：`184 passed in 83.32s`（2026-09-17）。

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
16. **GPU 并发上限（2026-09-17 用户要求）**：任何实验/agent 任务最多同时占用
    **4 张 GPU**；一次最多启动 4 个模型/训练/评测进程，多出的必须排队。写入命令
    前检查 `nvidia-smi` 与 `ps`，确认没有超额进程。

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
