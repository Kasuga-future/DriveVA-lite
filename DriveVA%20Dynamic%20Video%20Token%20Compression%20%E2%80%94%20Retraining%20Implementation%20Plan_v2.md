# DriveVA Dynamic Video Token Compression  
## 基于阈值动态压缩的重新训练实现方案

> **目标**：将 DriveVA 当前约 1560 个 video tokens 压缩为**动态少量 token**，并通过重新训练模型而非 inference-time hard prune，尽可能保持原始 DriveVA 的 trajectory planning 能力。  
>
> 本文只保留两条独立路线：
>
> 1. **Route A — Dynamic Select**：只保留原始 video patch token，不生成新的聚合 token。
> 2. **Route B — Dynamic Register**：将 dense video token 重新压缩为少量 latent/register token。
>
> 两条路线均使用：
>
> \[
> \boxed{\text{score threshold} \rightarrow \text{dynamic token count}}
> \]
>
> 不再使用固定 K 或 96/128/160/192 等离散分档。

---

# 1. 当前问题与实现边界

## 1.1 DriveVA 当前 token 布局

当前默认：

```text
history frames = 5
future frames  = 8
resolution     = 480 × 832
target FPS     = 2
```

VAE 后共有 4 个 latent frames：

```text
history latent: 2 × 390 = 780 tokens
future latent : 2 × 390 = 780 tokens

video total   : 1560 tokens
trajectory    : ~9 tokens

DiT sequence  : ~1569 tokens
```

该布局已经在当前工程中确认。

---

## 1.2 已验证结论

现有实验已经证明：

```text
Frozen DriveVA
    +
future hard prune
```

不能实现严格 near-lossless。

尤其是：

- future keep 50% 明显掉 PDM；
- keep 87.5% 仍未达到严格 near-lossless；
- L18/L22 selector 重训没有解决；
- selector 确实存在一定 selection signal，但不足以挽救 hard prune；
- 当前严格近无损部署仍主要依赖 history compression；
- future token 应停止按“模型已有冗余”继续寻找 free lunch。

因此新任务必须从：

```text
find useless tokens in existing DriveVA
```

改成：

```text
train DriveVA to represent the same driving information
with far fewer video tokens
```

---

# 2. 总体设计原则

## 2.1 不改变 Flow Matching 的原始 state space

第一版不要直接修改 VAE latent 或重新定义 Flow Matching distribution。

仍保持：

\[
Z_\sigma
=
(1-\sigma)Z_0+\sigma\epsilon
\]

以及 trajectory：

\[
A_\sigma
=
(1-\sigma)A_0+\sigma\epsilon_A
\]

其中：

```text
Zσ = dense video latent
Aσ = trajectory state
```

压缩发生在：

```text
VAE latent
    ↓
Patch Embedding
    ↓
dense video hidden
    ↓
[Compression]
    ↓
compact DiT sequence
```

而不是：

```text
VAE latent
    ↓
直接变成一个新的 flow state
```

原因是这样可以继续完全复用：

- `FlowMatchScheduler`
- 原 video FM target
- 原 trajectory FM target
- 原 trajectory normalization
- 原 inference scheduler

降低一次性改变模型过多组成部分的风险。

---

## 2.2 模型初始化策略：默认从 DriveVA，而不是 WAN2.2 开始

本项目的主目标不是重新训练一个新的自动驾驶模型，而是：

> **在尽可能保持原 DriveVA world-action / planning 能力的前提下，把 dense video representation 重新组织为动态少量 token。**

因此两条主路线的 **student 默认都从现有 DriveVA checkpoint 初始化**：

```text
Route A:
DriveVA checkpoint
    ↓
Dynamic Select
    ↓
compression-aware retraining

Route B:
DriveVA checkpoint
    ↓
Dynamic Register
    ↓
compression-aware retraining
```

而不是默认：

```text
WAN2.2
    ↓
重新学习 driving adaptation
    +
trajectory grounding
    +
token compression
```

这样做的原因是：DriveVA 已经完成了从通用视频生成模型到 Video-Action Model 的适配，已经学习了 video ↔ trajectory token 联合 attention、trajectory generative semantics、ego/command conditioning，以及 future video 与 future trajectory 的联合 Flow Matching。若直接从 WAN2.2 开始，训练同时需要解决：

\[
\text{Driving adaptation}+\text{Action grounding}+\text{Token compression}
\]

而从 DriveVA 开始主要需要解决：

\[
\text{Representation reorganization}+\text{Token compression}
\]

这使实验更容易收敛，也更容易把性能变化归因于 compression 本身。

### 2.2.1 DriveVA 同时作为初始化和 Teacher

主实验统一采用：

```text
Student initialization:
original DriveVA checkpoint

Teacher:
frozen original DriveVA / NoPress
```

因此压缩训练优化：

\[
f_{student}^{compressed}(x)\approx f_{DriveVA}(x)
\]

并同时接受 GT Flow Matching supervision。这也是后文 trajectory/video flow KD 与 action/video hidden KD 的依据。

### 2.2.2 为什么不把 WAN2.2 作为第一主线

当前 clean training split 只有约 3k 级别 scenes。该规模适合在已经完成 driving adaptation 的 DriveVA 上做 representation adaptation；若从 WAN2.2 重新开始，则很难区分：

```text
PDM 没恢复
```

究竟来自 compression architecture 失败，还是 WAN2.2 → DriveVA adaptation 没有重新训练充分。

因此主实验应优先保持：

\[
\Delta PDM\approx\text{compression / adaptation effect}
\]

而不是把能力恢复问题与压缩问题混在一起。

### 2.2.3 WAN2.2 的正确定位：representation-lock-in 对照

WAN2.2 仍然是重要的第二阶段实验。DriveVA 已经在 dense 1560-video-token 训练方式下形成内部表示和 attention pattern，可能存在：

\[
\boxed{\text{dense representation lock-in}}
\]

因此当 Route B 已经证明 Dynamic Register 架构可行之后，补如下初始化对照：

```text
A. DriveVA DiT
   + DriveVA trajectory encoder/head
   + Dynamic Register

B. WAN2.2 DiT
   + DriveVA trajectory encoder/head
   + Dynamic Register

C. Optional:
   WAN2.2 DiT
   + randomly initialized trajectory encoder/head
   + Dynamic Register
```

其中：

- **A**：主线，最容易保持原始 PDM；
- **B**：最重要的 ablation，不继承 DriveVA dense visual attention pattern，但保留 action interface；
- **C**：完全 compression-aware 地重新做 driving adaptation，只在资源允许时进行。

如果 B 在相同 token 数下明显优于 A，则支持这样一个研究解释：

> **future token 难压的一部分原因来自 DriveVA dense-token fine-tuning 形成的 representation specialization，而不一定是驾驶任务本身需要高维 dense visual state。**

---

# 3. 阈值动态压缩统一定义

两种路线都不要直接预测 K。

每个候选 token/register 输出：

\[
l_i=f_\phi(h_i,c)
\]

其中 condition \(c\) 可以包括：

```text
action hidden
ego velocity
flow timestep σ
denoising round
token type
temporal position
spatial position
```

归一化为：

\[
s_i=\sigma(l_i)\in(0,1)
\]

部署时：

\[
m_i=
\mathbf 1(s_i\ge\tau)
\]

最终长度：

\[
K(x)=\sum_i m_i
\]

所以：

\[
\boxed{K=K(scene,\sigma,round)}
\]

是自然动态变化的。

---

# 4. 阈值训练方式

hard threshold 本身不可微，因此训练时采用 Straight-Through Estimator。

```python
logits = scorer(hidden, context)

scores = torch.sigmoid(logits)

soft_mask = torch.sigmoid(
    (scores - threshold) / temperature
)

hard_mask = (scores >= threshold).float()

mask = hard_mask.detach() \
     - soft_mask.detach() \
     + soft_mask
```

forward：

```text
hard threshold
```

backward：

```text
soft sigmoid approximation
```

训练后 inference 完全不需要 soft mask：

```python
keep = scores >= threshold
```

---

# 5. 不再使用固定 retention target

不要训练：

\[
K=128
\]

或者：

\[
K/N=10\%
\]

这种固定预算。

改成 Lagrangian sparsity：

\[
L_{sparse}
=
\frac1N\sum_i s_i
\]

总目标：

\[
L
=
L_{task}
+
\lambda_{sparse}L_{sparse}
\]

含义：

> 只要删除 token 不影响 task，模型就会受到压力降低其 score；一旦 token 对任务重要，就必须提高 score 越过 threshold。

最终平均压缩率是训练结果，而不是预先硬编码的 K。

---

# 6. Threshold 的推荐形式

第一版采用**固定阈值、动态长度**。

建议分别维护：

```text
tau_history
tau_future
```

因为 history/future hidden 的统计分布不同。

例如：

```python
keep_history = score_history >= tau_history
keep_future  = score_future  >= tau_future
```

不要强制：

```text
history keep 64
future keep 128
```

---

## 6.1 Safety clamp

允许保留极宽松的保护范围：

```python
K_min <= K_dynamic <= K_max
```

例如：

```text
history Kmin = 8
future  Kmin = 32

global Kmax = 384
```

这些不是压缩档位，只是避免训练初期：

```text
所有 token 被删
```

或：

```text
所有 token 全保留
```

导致数值不稳定。

模型稳定后可以取消 Kmax。

---

# 7. Route A — Dynamic Select

## 7.1 核心思想

Route A 不生成新的 video representation。

只能从已有：

\[
V=\{v_1,\dots,v_{1560}\}
\]

中选择一个 subset：

\[
V_S=\{v_i|s_i\ge\tau\}
\]

所以 compact sequence 中所有 video tokens 都仍是：

```text
original spatial patch tokens
```

---

# 8. Route A 模型结构

```text
dense noisy video latent
        │
        ▼
    Patch Embedding
        │
   1560 video tokens
        │
        ▼
Dense DiT Block 0 ... Lb-1
        │
        ▼
 Threshold Scorer
        │
        │ score >= τ
        ▼
dynamic selected tokens
        │
        + trajectory tokens
        │
        ▼
Sparse DiT Block Lb ... 29
        │
        ├─────────────► Trajectory Head
        │
        ▼
Dense Recovery Decoder
        │
        ▼
1560 video hidden
        │
        ▼
original Wan Head
        │
        ▼
video flow prediction
```

---

# 9. 为什么 Route A 不能在 L0 直接 select

已有分析显示：

- early hidden 更接近 VAE patch texture；
- history semantic 大约在 L8–15 逐渐形成；
- future representation 更晚，在约 L16–18 稳定；
- trajectory semantic 大约在 L10–12 已形成。

因此原模型：

```text
L0
 ↓
selector
```

很难知道某个 patch 在 20 层之后是否重要。

Route A 应采用 compression-layer curriculum。

建议：

```text
初始：Lb = 18

稳定后：
Lb = 15

进一步训练：
Lb = 12
```

第一篇主要实验不建议低于：

```text
Lb = 10
```

---

# 10. Route A Scorer

建议模块：

```python
class DynamicVideoTokenScorer(nn.Module):

    def __init__(self, dim):
        self.token_proj  = MLP(dim, dim // 4, 1)
        self.action_proj = MLP(dim, dim // 4)
        self.time_proj   = MLP(time_dim, dim // 4)
        self.pos_proj    = MLP(pos_dim, dim // 4)

    def forward(
        self,
        video_hidden,
        action_hidden,
        timestep,
        positions,
        token_type,
    ):

        action_context = action_hidden.mean(dim=1)

        context = (
            self.action_proj(action_context)
            + self.time_proj(timestep)
        )

        token_feature = (
            self.token_proj(video_hidden)
            + self.pos_proj(positions)
        )

        logits = interaction(
            token_feature,
            context
        )

        return logits
```

关键是 scorer 必须知道：

```text
current action state
flow timestep
position
history / future identity
```

不能再只从 video token 本身判断重要性。

---

# 11. Route A Forward 伪代码

```python
# ==================================================
# Flow Matching inputs
# ==================================================

z0 = vae.encode(gt_video)
a0 = normalize(gt_trajectory)

sigma = sample_flow_timestep()

eps_v = randn_like(z0)
eps_a = randn_like(a0)

z_t = (1 - sigma) * z0 + sigma * eps_v
a_t = (1 - sigma) * a0 + sigma * eps_a

target_video = eps_v - z0
target_action = eps_a - a0


# ==================================================
# Dense frontend
# ==================================================

V = patchify(z_t)

A = trajectory_encoder(
    a_t,
    velocity=ego_velocity,
)

X = concat(V, A)

for layer in range(L_bottleneck):
    X = dit.blocks[layer](X)


V, A = split_video_action(X)


# ==================================================
# Threshold selector
# ==================================================

logits = selector(
    video_hidden=V,
    action_hidden=A,
    timestep=sigma,
    positions=video_positions,
    token_type=history_future_flag,
)

scores = sigmoid(logits)

mask_h = scores[history] >= tau_history
mask_f = scores[future]  >= tau_future

mask = concat(mask_h, mask_f)

mask = safety_clamp(
    mask,
    scores,
    min_tokens=K_min,
    max_tokens=K_max,
)

V_sparse = V[mask]

sparse_rope = video_rope[mask]


# ==================================================
# Sparse backbone
# ==================================================

X = concat(V_sparse, A)

for layer in range(L_bottleneck, NUM_BLOCKS):

    X = dit.blocks[layer](
        X,
        freqs=concat(
            sparse_rope,
            trajectory_rope
        ),
    )


V_sparse_out, A_out = split_video_action(X)


# ==================================================
# Trajectory prediction
# ==================================================

pred_action_flow = trajectory_head(A_out)


# ==================================================
# Recover dense video hidden
# Training only / optional inference
# ==================================================

V_dense_out = dense_recovery_decoder(
    sparse_tokens=V_sparse_out,
    sparse_positions=mask,
    full_query_positions=video_positions,
)

pred_video_flow = video_head(V_dense_out)

pred_video_flow = unpatchify(pred_video_flow)
```

---

# 12. Route A Dense Recovery Decoder

因为 Flow Matching 的 video target 仍然是 dense：

\[
v_V\in\mathbb R^{1560}
\]

selected tokens 必须恢复回 full grid。

推荐使用 1–2 层 query decoder：

\[
Q=P_{full}
\]

\[
K,V=H_{selected}
\]

```python
dense_hidden = CrossAttention(
    query=full_position_queries,
    key=sparse_hidden,
    value=sparse_hidden,
)
```

作用仅是：

```text
compact representation
    ↓
recover dense video flow
```

它不参与主 DiT reasoning。

trajectory-only inference 可以完全关闭该 decoder。

---

# 13. Route A Loss

主要 loss：

\[
L_{A}=
L_{trajFM}
+\lambda_vL_{videoFM}
+\lambda_{trajKD}L_{trajKD}
+\lambda_{hidden}L_{actionHiddenKD}
+\lambda_sL_{sparse}
\]

推荐：

```text
trajectory FM       1.0
video FM            1.0
trajectory KD       2.0
action hidden KD    1.0
video KD            0.5
sparsity            curriculum
```

其中：

\[
L_{sparse}
=
mean(score)
\]

---

# 14. Route A 训练阶段

## A0 — Dense teacher cache + Student 初始化

Teacher：

```text
frozen original DriveVA / NoPress
```

Student：

```text
initialize from the same DriveVA checkpoint
```

原则是先继承完整 DriveVA planning function，再学习把该 function 重组织到少量 token。Teacher 全程冻结。

缓存关键层：

```text
trajectory flow output
video flow output
action hidden @ L11
action hidden @ L18
action hidden @ L29
```

---

## A1 — Selector warmup

冻结：

```text
DriveVA backbone
```

训练：

```text
selector
dense recovery decoder
```

暂时设置：

```text
tau 较低
lambda_sparse 很小
```

目标是让模型先学会：

```text
score 与重要性相关
```

而不是立即追求极限压缩。

---

## A2 — LoRA adaptation

训练：

```text
selector
decoder
trajectory head
DiT LoRA
```

推荐：

```text
rank = 64
```

使原始 DiT 开始主动将信息写入可能被保留的 tokens。

---

## A3 — Full DiT adaptation

训练：

```text
dit
trajectory_encoder
trajectory_head
selector
decoder
```

此时逐渐提高：

\[
\lambda_{sparse}
\]

让平均动态长度下降。

---

## A4 — Earlier compression

如果 L18 near-lossless：

```text
L18
 ↓
L15
 ↓
L12
```

每移动一次 bottleneck layer 都重新 fine-tune。

不要直接从：

```text
L18 → L8
```

---

# 15. Route A 推荐压缩目标

Route A 不设置固定 K。

但训练期希望最终统计达到：

```text
average video retention:
12% ~ 18%
```

即平均大约：

\[
190\sim280
\]

video tokens。

注意：

```text
简单场景可能只有 100~150
复杂场景可能 300+
```

这是正常现象。

最终应该报告：

```text
mean
median
P10
P50
P90
P95
```

而不能只报告平均 K。

---

# 16. Route B — Dynamic Register Compression

## 16.1 核心思想

Route B 不保留原 patch token。

而是学习：

\[
1560\ dense\ tokens
\rightarrow
M(x)\ register\ tokens
\]

其中：

\[
M(x)
\]

也是由 threshold 决定的动态数目。

---

# 17. 为什么不能复用旧 RegisterBottleneck

旧实验已经证明简单：

```text
global cross-attention
        ↓
few registers
        ↓
LoRA adaptation
```

严重掉 PDM。

新设计必须改变两个地方：

1. Register 从训练开始就是模型的 native representation。
2. Register 使用**结构化局部重压缩**，而不是所有 query 全局平均所有 patch。

---

# 18. Route B 模型结构

```text
dense noisy latent
       │
       ▼
 Patch Embedding
       │
 1560 patches
       │
       ▼
Structured Resampler
       │
       │ Mmax register candidates
       ▼
Register Scorer
       │
   score >= tau
       ▼
dynamic register set
       │
       + trajectory tokens
       │
       ▼
 Compact DiT × 30
       │
       ├────────────► trajectory head
       │
       ▼
Dense Query Decoder
       │
       ▼
1560 dense hidden
       │
       ▼
 original Wan Head
       │
       ▼
video flow
```

与 Route A 最大区别：

```text
Route A:
selected token = original patch

Route B:
register = learned aggregation of many patches
```

---

# 19. Structured Resampler

不要让所有 register attend 所有 1560 tokens。

建议定义多个 spatial-temporal regions。

例如每个 latent frame：

```text
15 × 26 patch grid
```

划分为粗网格：

```text
4 × 8 regions
```

每个 region 有若干 candidate registers：

```text
region
 ├─ register 0
 └─ register 1
```

每个 register：

\[
r_j=
CrossAttention(
q_j,
V_{\Omega_j}
)
\]

其中：

\[
\Omega_j
\]

只是 local neighborhood。

因此局部语义不会被整张图平均。

---

# 20. Register candidate bank

定义最大 candidate bank：

```text
Mmax = 256
```

注意：

\[
256
\]

只是 candidate 上限，不是最终 token 数。

每个 register 生成以后得到：

\[
r_j
\]

再计算：

\[
s_j=f_\phi(r_j,A,\sigma,pos)
\]

部署：

\[
r_j\ kept
\iff
s_j\ge\tau_R
\]

因此：

```text
scene A → 91 registers
scene B → 137 registers
scene C → 203 registers
```

没有任何固定档位。

---

# 21. Route B Forward 伪代码

```python
# ==================================================
# Original Flow Matching state
# ==================================================

z0 = vae.encode(gt_video)
a0 = normalize(gt_trajectory)

sigma = sample_flow_timestep()

eps_v = randn_like(z0)
eps_a = randn_like(a0)

z_t = (1 - sigma) * z0 + sigma * eps_v
a_t = (1 - sigma) * a0 + sigma * eps_a

target_video = eps_v - z0
target_action = eps_a - a0


# ==================================================
# Dense patch representation
# ==================================================

V_dense = patchify(z_t)

A = trajectory_encoder(
    a_t,
    velocity=ego_velocity,
)


# ==================================================
# Build register candidates
# ==================================================

registers = []

for region in spatiotemporal_regions:

    V_local = gather_region(
        V_dense,
        region,
    )

    R_local = local_resampler(
        learned_queries[region],
        V_local,
        timestep=sigma,
    )

    registers.append(R_local)

R_candidates = concat(registers)


# ==================================================
# Register scoring
# ==================================================

register_logits = register_scorer(
    registers=R_candidates,
    action_hidden=A,
    timestep=sigma,
    positions=register_positions,
)

register_scores = sigmoid(register_logits)

register_mask = (
    register_scores >= tau_register
)

register_mask = safety_clamp(
    register_mask,
    register_scores,
    min_tokens=K_min,
    max_tokens=K_max,
)

R = R_candidates[register_mask]


# ==================================================
# Compact DriveVA
# ==================================================

X = concat(R, A)

for block in dit.blocks:

    X = block(
        X,
        freqs=concat(
            register_rope[register_mask],
            trajectory_rope,
        ),
    )


R_out, A_out = split_register_action(X)


# ==================================================
# Trajectory
# ==================================================

pred_action_flow = trajectory_head(A_out)


# ==================================================
# Dense reconstruction
# ==================================================

V_dense_out = dense_video_decoder(
    query=full_video_queries,
    key=R_out,
    value=R_out,
)

pred_video_flow = video_head(
    V_dense_out
)

pred_video_flow = unpatchify(
    pred_video_flow
)
```

---

# 22. Route B register score

推荐：

```python
score_j = MLP(
    concat(
        register_j,
        pooled_action_hidden,
        timestep_embedding,
        register_position_embedding,
    )
)
```

不要仅使用：

```python
||register||
```

或者纯 attention weight。

因为 score 的目标是：

```text
planning usefulness
```

而不是 register magnitude。

---

# 23. Route B Loss

\[
L_B=
L_{trajFM}
+
L_{videoFM}
+
L_{trajKD}
+
L_{actionHiddenKD}
+
L_{videoKD}
+
L_{registerSparse}
\]

其中：

\[
L_{registerSparse}
=
\frac1{M_{max}}
\sum_j s_j
\]

同样不使用固定 register count target。

---

# 24. Route B 训练阶段

## B0 — Dense hidden autoencoding

Student 默认从 **DriveVA checkpoint** 初始化。

这一阶段冻结 DriveVA backbone，只训练新增 compression modules；不要从 WAN2.2 初始化主实验，否则会把 driving adaptation 与 bottleneck learning 混在一起。

只训练：

```text
structured resampler
dense decoder
```

目标：

\[
V_{dense}
\rightarrow
R
\rightarrow
\hat V_{dense}
\]

先验证：

> 这些 register 是否至少能够承载 dense hidden 中的信息。

---

## B1 — Dense teacher distillation

Teacher：

```text
frozen original DriveVA / NoPress
```

Student：

```text
DriveVA-initialized Register DriveVA
```

即：

```text
DriveVA checkpoint
    +
Structured Resampler
    +
Register Scorer
    +
Dense Decoder
```

而不是从 WAN2.2 重新开始。

主要训练：

```text
resampler
register scorer
dense decoder
trajectory head
DiT LoRA
```

---

## B2 — Full adaptation

解除 DiT。

训练：

```text
resampler
register scorer
DiT
trajectory encoder
trajectory head
dense decoder
```

---

## B3 — Sparsity curriculum

开始：

```text
lambda_sparse ≈ 0
```

然后缓慢增加：

```python
lambda_sparse = ramp(
    start_step,
    end_step,
    lambda_max
)
```

随着模型学习将信息集中到 register 中：

```text
score distribution
```

应该逐渐变得两极化：

```text
important → score → 1
redundant → score → 0
```

动态 threshold 才会稳定。

---

# 25. Threshold calibration

训练完成以后，不应该根据训练集选择 threshold。

使用独立 calibration split。

项目已有：

```text
train       = 3190 scenes
calibration = 578 scenes
```

且 train/calibration/test 没有 scene overlap。

对 calibration 做：

```text
tau = 0.20
tau = 0.25
tau = 0.30
...
tau = 0.80
```

记录：

```text
PDM
ΔPDM
CI
mean token count
P50 count
P90 count
latency
zero-score count
```

最终选择满足质量约束下压缩最多的 threshold。

---

# 26. Near-lossless 判定标准

继续沿用项目现有严格标准：

\[
\boxed{
CI_{lower}(\Delta PDM)>-0.002
}
\]

不要因为：

```text
mean ΔPDM = -0.001
```

就判定成功。

必须同时报告：

```text
mean delta
95% CI
introduced zero scenes
rescued scenes
extreme flips
```

当前项目已经多次观察到：

```text
平均 PDM 接近
```

但尾部 zero-score 场景明显恶化的问题，因此 tail safety 必须单独检查。

---

# 27. Training Dataset

第一阶段只使用现有 NAVSIM training split。

```text
train:
3190 scenes

calibration:
578 scenes

official evaluation:
7876 scenes
```

训练输入保持：

```text
height = 480
width  = 832

history frames = 5
future frames  = 8

FPS = 2

trajectory condition = velocity
```

不要在第一版把：

```text
nuScenes
Bench2Drive
```

加入训练。

它们应保留做：

```text
OOD / zero-shot preservation
```

测试。

---

## 27.1 Initialization Ablation Protocol

主实验固定：

```text
Init-A:
DriveVA DiT
+ DriveVA TrajectoryEncoder
+ DriveVA TrajectoryHead
```

两条路线都先用 Init-A。只有当 Route B 已经达到可工作的 near-lossless 区域后，再做：

```text
Init-B:
WAN2.2 DiT
+ DriveVA TrajectoryEncoder
+ DriveVA TrajectoryHead
```

控制变量必须保持一致：

```text
same Dynamic Register architecture
same threshold mechanism
same NAVSIM train split
same optimizer / LR budget
same number of update steps
same FM / KD supervision
same calibration protocol
same 7,876-scene paired evaluation
```

Init-B 不是“失败后换初始化”，而是用于检验：

\[
\text{dense DriveVA specialization}
\quad vs \quad
\text{task-intrinsic information requirement}
\]

如果资源充足，再增加：

```text
Init-C:
WAN2.2 DiT
+ randomly initialized trajectory modules
```

Init-C 只研究完全 compression-aware 的 DriveVA adaptation，不作为主工程路线。

### Route-specific recommendation

```text
Dynamic Select:
    必须优先 DriveVA-init。
    不建议把 WAN2.2-init 作为主实验。

Dynamic Register:
    第一阶段 DriveVA-init。
    架构验证成功后，优先补
    WAN2.2 DiT + DriveVA trajectory modules
    作为 representation-lock-in ablation。
```

---

# 28. 推荐训练参数

## Shared

```yaml
precision: bf16

optimizer: AdamW
weight_decay: 0.01

gradient_checkpointing: true
grad_clip: 1.0

ema: true
ema_decay: 0.999

micro_batch_per_gpu: 1
gradient_accumulation: 8-16

save_steps: 250-500
```

---

## New modules

适用于：

```text
selector
register resampler
register scorer
dense decoder
```

推荐：

```yaml
lr: 1.0e-4
```

---

## LoRA adaptation

```yaml
rank: 64

target:
  - q
  - k
  - v
  - o
  - ffn.0
  - ffn.2

lr: 5.0e-5 ~ 1.0e-4
```

---

## Full DiT adaptation

不要继续使用新模块的：

```text
1e-4
```

建议：

```yaml
dit_lr: 5.0e-6 ~ 1.0e-5

trajectory_encoder_lr: 2.0e-5

trajectory_head_lr: 2.0e-5

compression_module_lr: 5.0e-5 ~ 1.0e-4
```

建议 optimizer param groups 分开设置。

---

# 29. Threshold training 参数

建议初始：

```yaml
score_temperature:
  start: 0.20
  end: 0.05

threshold_history: 0.5
threshold_future: 0.5

threshold_register: 0.5
```

训练时可随机扰动：

```python
tau_train = tau_base + Uniform(-0.05, 0.05)
```

目的不是产生 K 档位，而是避免模型只适配单一 threshold。

部署仍使用单一校准后的：

```text
tau*
```

---

# 30. Sparsity weight curriculum

不要第一 step 就强压。

例如：

```python
if step < warmup:
    lambda_sparse = 0

elif step < sparse_ramp_end:
    lambda_sparse = linear_ramp(
        0,
        lambda_sparse_max
    )

else:
    lambda_sparse = lambda_sparse_max
```

推荐初次 sweep：

```text
1e-4
3e-4
1e-3
3e-3
1e-2
```

最终通过：

```text
PDM-retention Pareto
```

决定。

---

# 31. Teacher Distillation

Teacher 必须保持：

```text
NoPress original DriveVA
```

而不是当前 Press checkpoint。

至少 distill：

```text
final trajectory flow
final video flow
action hidden L11
action hidden L18
action hidden L29
```

推荐：

\[
L_{actionHidden}
=
\sum_{l\in\{11,18,29\}}
\|
LN(H^S_{A,l})
-
LN(H^T_{A,l})
\|_2^2
\]

原因：

> 目标不是仅仅重新训练一个能达到相似 PDM 的模型，而是尽量保留原 DriveVA 的 planning function。

---

# 32. 建议的总 Loss

第一版：

\[
\begin{aligned}
L=&
1.0L_{trajFM}\\
&+1.0L_{videoFM}\\
&+2.0L_{trajKD}\\
&+0.5L_{videoKD}\\
&+1.0L_{actionHiddenKD}\\
&+0.5L_{videoHiddenKD}\\
&+\lambda_sL_{sparse}
\end{aligned}
\]

其中：

```text
Route A:
Lsparse = mean(video token scores)

Route B:
Lsparse = mean(register scores)
```

---

# 33. 推荐压缩目标

虽然 inference 不设固定 K，但需要给研究设定预期工作区间。

## Route A

预期：

```text
mean retention ≈ 12% ~ 18%
```

对应平均：

```text
~190-280 video tokens
```

重点不是平均数，而是：

```text
easy scene   → 更少
normal scene → 中等
hard scene   → 更多
```

---

## Route B

目标可以更激进：

```text
mean retention ≈ 6% ~ 12%
```

即平均：

```text
~90-190 registers
```

推荐第一篇工作目标：

\[
\boxed{
mean\approx128\sim160
}
\]

但不能通过 hard K 实现。

应该是：

```text
fixed threshold
     ↓
scene-dependent M
```

---

# 34. Runtime statistics 必须记录

每个 scene、每个 inference round 保存：

```json
{
  "scene_id": "...",
  "round": 1,
  "sigma": 0.87,

  "history_candidates": 780,
  "future_candidates": 780,

  "history_kept": 83,
  "future_kept": 146,

  "total_video_kept": 229,

  "score_mean": 0.31,
  "score_std": 0.22,

  "score_p10": 0.05,
  "score_p50": 0.24,
  "score_p90": 0.72,

  "threshold": 0.50
}
```

Route B 改为：

```text
register_candidates
register_kept
```

---

# 35. 必须分析动态长度是否真的“动态”

最终报告至少计算：

\[
corr(K,PDM\ difficulty)
\]

以及：

\[
corr(K,\sigma)
\]

并按 scene category 统计：

```text
straight
turn
intersection
traffic light
high traffic
low traffic
```

我们希望观察到：

```text
简单场景 → 少 token
困难场景 → 多 token
```

如果最后：

```text
所有 scene 都保留差不多 150
```

说明 threshold scorer 实际退化成了固定-budget selector。

---

# 36. 代码实现位置

当前项目核心代码地图已经明确：

- `diffsynth/pipelines/wan_video_new.py`
- `videopress_framework/videopress/core/`
- `videopress_framework/videopress/training/`
- `examples/wanvideo/driveva_train/`
- `examples/wanvideo/driveva_infer/`

不要把最终 retrained model 实现成 runtime Press hook。

现有：

```text
BLOCK_INPUT
SELF_ATTN_KV
hidden_sequence
```

继续保留做：

```text
probe
baseline
ablation
```

但新 compression 应成为：

```text
pipe 的正式 nn.Module
```

并写入 checkpoint。

---

# 37. 推荐新增代码结构

```text
videopress_framework/
└── videopress/
    └── retraining/
        ├── __init__.py
        │
        ├── threshold_gate.py
        │
        ├── dynamic_selector.py
        │
        ├── dense_recovery.py
        │
        ├── structured_resampler.py
        │
        ├── register_scorer.py
        │
        ├── distillation.py
        │
        └── compression_stats.py
```

训练：

```text
examples/wanvideo/driveva_train/

├── train_navsim_v1.py
├── train_navsim_dynamic_select.py
└── train_navsim_dynamic_register.py
```

---

# 38. Pipeline 修改建议

`WanVideoPipeline` 新增：

```python
self.video_compressor = None
self.video_decompressor = None
self.video_token_scorer = None

self.compression_mode = "none"
```

支持：

```text
none
dynamic_select
dynamic_register
```

---

# 39. model_fn 推荐接口

```python
def model_fn_wan_video(
    ...,
    video_compressor=None,
    video_decompressor=None,
    video_token_scorer=None,
    compression_mode="none",
    compression_threshold=None,
    return_compression_stats=False,
):
```

不要把训练逻辑隐藏在 runtime hook。

---

# 40. Route A 插入点

```python
x = dit.patchify(x)

x = flatten(x)

for i, block in enumerate(dit.blocks):

    x = block(...)

    if (
        compression_mode == "dynamic_select"
        and i == bottleneck_layer
    ):
        x, compression_state = (
            video_compressor(...)
        )
```

---

# 41. Route B 插入点

Route B 在进入 Block 0 前：

```python
x = dit.patchify(x)

x = flatten(x)

if compression_mode == "dynamic_register":

    x, compression_state = (
        video_compressor(...)
    )

for block in dit.blocks:
    x = block(...)
```

因此全部 30 层获得 sequence reduction。

---

# 42. Inference

## Route A

```text
dense frontend
→ threshold
→ sparse backend
→ trajectory head
```

如果：

```text
--infer-trajectory-only
```

则：

```text
Dense Recovery Decoder
```

直接跳过。

---

## Route B

```text
patchify
→ register resampler
→ threshold registers
→ compact DiT
→ trajectory head
```

trajectory-only inference 同样跳过 dense video decoder。

---

# 43. 实验推进顺序

### Experiment 1

```text
Route A
Lb = 18
低 sparsity pressure
```

问题：

> retraining 能否第一次让 future compression 超过 frozen hard-prune 上界？

---

### Experiment 2

```text
Route A
Lb = 15
```

验证是否可以进一步提前。

---

### Experiment 3

```text
Route B
Mmax = 256
低 sparsity pressure
```

先确认 register model 能恢复 teacher planning。

---

### Experiment 4

逐渐增加：

\[
\lambda_{sparse}
\]

观察：

```text
PDM vs mean dynamic length
```

---

### Experiment 5

calibration threshold sweep。

---

### Experiment 6

最终 full 7,876 paired evaluation。

官方主协议当前 7,876 scenes，NoPress baseline 与评测路径已经在工程中固定。

---

### Experiment 7 — Initialization / Representation-Lock-in Ablation

仅当 Route B 的 DriveVA-init 版本已经证明架构可工作后进行。

固定 Route B 架构与训练预算，对比：

```text
A. DriveVA DiT
   + DriveVA trajectory modules
   + Dynamic Register

B. WAN2.2 DiT
   + DriveVA trajectory modules
   + Dynamic Register
```

主要比较：

```text
PDM / ΔPDM / 95% CI
mean register count
P50 / P90 register count
latency
zero-score tail
threshold-retention frontier
```

解释：

```text
A 更好：
    继承 DriveVA driving representation 更重要，
    dense representation lock-in 不是主要瓶颈。

B 在相同 token 数下更好：
    DriveVA dense-token adaptation 可能形成了
    不利于极限压缩的 representation specialization。
```

如果资源充足，再加入：

```text
C. WAN2.2 DiT
   + random trajectory modules
```

用于研究完全 compression-aware driving adaptation。

---

# 44. Go / No-Go 标准

## Route A 成功标准

至少满足：

```text
CI lower bound > -0.002

AND

mean video tokens < 300
```

否则 Select Route 研究价值有限。

---

## Route B 成功标准

第一阶段：

```text
CI lower bound > -0.002

AND

mean registers < 256
```

正式目标：

```text
mean registers < 160
```

stretch goal：

```text
mean registers < 128
```

---

# 45. 两条路线与初始化的优先级

推荐执行顺序：

```text
1. DriveVA-init Route A — Dynamic Select
2. DriveVA-init Route B — Dynamic Register
3. WAN2.2 DiT + DriveVA trajectory modules
   Route B initialization ablation
4. Optional full WAN2.2-init Route B
```

原因不是认为 Route A 最终更强，而是 Route A 改动更小、DriveVA-init 的因果解释最干净。

它首先回答：

> 当 backbone 被允许重新训练以后，现有 future-token hard-prune 的失败是否主要来自 representation mismatch？

若答案为 YES，说明重新训练方向成立。

随后 DriveVA-init Route B 回答：

> aggregation 是否能把 representation 从约 200 个原 patch token 进一步压到约 100 个 latent registers？

最后 WAN2.2-based Route B 不再作为同级主线，而是专门回答：

> 原 DriveVA 在 dense-token fine-tuning 中是否形成了限制进一步压缩的 representation lock-in？

---

# 46. 最终研究问题

整个项目最终不再表述为：

> “DriveVA 中哪些 video token 可以删除？”

而应改成：

> **Can a jointly trained DriveVA reorganize dense video representations into a dynamically sparse set of planning-sufficient tokens while preserving its original world-action behavior?**

对应两条技术答案：

```text
Dynamic Select:
Can information be reorganized into a small subset
of original spatial tokens?

Dynamic Register:
Can dense visual dynamics be recompressed into
a small learned latent memory?
```

两者统一由：

\[
\boxed{
score_i\ge\tau
}
\]

控制最终动态 token 数量，而不是人工预设固定 token budget。

此外，模型初始化明确分为：

```text
Mainline:
DriveVA-init

Secondary scientific ablation:
WAN2.2 DiT + DriveVA trajectory modules

Optional:
full WAN2.2-init
```

从而把两个问题分开研究：

1. **Compression problem**：已经学会 driving 的 DriveVA 能否把 dense representation 重组为动态稀疏 token？
2. **Representation-lock-in problem**：若从 WAN2.2 阶段就引入 compression-aware representation，是否能获得更高的极限压缩率？

主工程目标优先回答第一个问题；第二个问题作为 Route B 成功后的关键 ablation。