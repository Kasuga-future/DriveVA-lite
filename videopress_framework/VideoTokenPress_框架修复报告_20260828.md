# DriveVA VideoTokenPress 框架修复报告

**日期**：2026-08-28  
**审查对象**：`videopress_framework.zip`  
**目标**：将当前 VideoTokenPress 从“可运行的独立框架原型”修复为“可用于 DriveVA 正式 token selection / compression / NAVSIM benchmark 的可信测试框架”。

---

## 1. 总体结论

当前代码已经完成了较好的框架抽象，核心结构：

```text
Domain
+ Scorer
+ Selector
+ Operator
+ Budget
+ InjectionPoint
```

建议全部保留，不需要推翻重写。

当前主要问题不在抽象层，而在 **真实 DriveVA 执行链路尚未闭环**。当前代码已经具备：

- `TokenLayout / TokenDomain / TokenBudget / TokenContext`
- `ScorerPress`
- Random / Norm / Attention / Gradient scorer
- Top-K / Threshold selector
- Zero / Mean / Shuffle / KVPrune / Merge operator
- Synthetic evaluator
- artifact/statistics 基础设施
- DriveVA post-RoPE attention hook 原型

但正式 benchmark 前必须修复以下问题：

1. `NoPress` artifact 写入会崩溃；
2. Merge mapping artifact 不会被保存；
3. DriveVA Adapter 实际忽略 `press.domain`；
4. `SimilarityMergePress` 在真实 DriveVA attention 中实际上是 no-op；
5. Gradient objective 的 target 读取路径不可靠，存在 silent objective degradation；
6. `_call_with_context()` 会吞掉函数内部真实 `TypeError`；
7. `VIDEO_INPUT` causal injection 尚未真正接入 DriveVA；
8. Adapter 没有按 `InjectionPoint` 路由；
9. 当前 latency 只测 press 本身，不是 DriveVA inference latency；
10. official NAVSIM evaluator 尚未接入；
11. `random_baseline` 配置目前未执行；
12. Random mask 依赖 `batch_index`，batch packing 改变会改变 mask；
13. `ActionAttention × ||V||` 在 single-head 模式下 V norm 定义与单 head 不严格匹配；
14. SimilarityMerge 当前 batch>1 时共享 merge plan；
15. `ShuffleOperator` 完全忽略 K/selection；
16. `ScoreCache` 尚未真正接入 probe/intervention 两遍执行；
17. Runtime 只保存 `last_result`，不能可靠记录 layer × diffusion-step 的压缩事件；
18. `mode=causal/physical` 尚未强制约束 operator/injection 语义。

因此当前阶段建议标记为：

```text
Core abstraction             PASS
Synthetic benchmark          PASS
Basic scorers/selectors      PASS
Causal operators standalone  PASS
DriveVA post-RoPE KV prune   PARTIAL PASS
DriveVA causal integration   NOT COMPLETE
DriveVA merge                FAIL / NO-OP
Gradient real integration    NOT COMPLETE
Official NAVSIM evaluation   NOT COMPLETE
Efficiency benchmark         NOT VALID YET
```

---

# 2. 修复优先级

## P0：必须修复后才能跑正式实验

| ID | 问题 | 风险 | 优先级 |
|---|---|---|---|
| P0-1 | NoPress artifact 崩溃 | Full baseline 无法统一走新 evaluator | Critical |
| P0-2 | Adapter 忽略 `press.domain` | 实验配置与真实 intervention domain 不一致，属于 silent scientific error | Critical |
| P0-3 | Gradient objective target 路径不统一 | Gradient ranking 可能优化错误 objective | Critical |
| P0-4 | `VIDEO_INPUT` 未真实接入 | Attention/Gradient/Random causal test 无法在真实 DriveVA 复现 | Critical |
| P0-5 | Adapter 不按 InjectionPoint 路由 | 配置语义与真实执行路径不一致 | Critical |
| P0-6 | SimilarityMerge 在 DriveVA 中 no-op | 会产生“压缩成功但模型实际没压”的假结果 | Critical |
| P0-7 | latency 测量范围错误 | speedup 数据无效 | Critical |
| P0-8 | official NAVSIM 未接入 | 当前 PDM 不是正式 DriveVA benchmark | Critical |

## P1：正式比较 selector 前必须修复

| ID | 问题 | 风险 |
|---|---|---|
| P1-1 | `random_baseline` 未实现 | 无法自动保证 equal-budget control |
| P1-2 | Random seed 包含 batch_index | batch 重排改变 mask |
| P1-3 | ScoreCache 未接两遍推理 | Attribution ranking 与 intervention 无持久化冻结保证 |
| P1-4 | Runtime 只保存 `last_result` | 多 layer / 多 diffusion step artifact 丢失 |
| P1-5 | Attention VNorm single-head 不严格匹配 | 无法精确复现旧 L15/H22 `A·||V||` 定义 |
| P1-6 | Shuffle 忽略 K | 配置显示不同 budget，实际 intervention 相同 |
| P1-7 | mode 不约束 operator | causal / physical protocol 可能混用 |

## P2：扩展 merge / batch /复杂方法前修复

| ID | 问题 |
|---|---|
| P2-1 | Merge artifact 与 selection artifact 耦合 |
| P2-2 | SimilarityMerge 对 batch 求均值，共享 merge plan |
| P2-3 | token index decode 缺少 video-range 校验 |
| P2-4 | Runtime sample artifacts 未明确 reset |
| P2-5 | unsupported injection point 应 fail-fast |

---

# 3. P0-1：修复 NoPress artifact 崩溃

## 3.1 当前问题

文件：

```text
evaluation/artifacts.py
```

当前逻辑：

```python
if result.selection is None:
    return

scores = result.scores.detach().cpu() if result.scores is not None else None
selected = result.selection.keep_global_indices.detach().cpu()

for batch_index in range(scores.shape[0]):
    ...
```

而 `NoPress` 返回：

```python
selection != None
scores = None
```

因此会访问：

```python
scores.shape
```

导致：

```text
AttributeError: 'NoneType' object has no attribute 'shape'
```

## 3.2 修复原则

ArtifactWriter 必须允许以下四类结果独立存在：

```text
selection only
scores only
mapping only
selection + scores + mapping
```

不能相互依赖。

## 3.3 推荐修改

```python
def add_result(self, ctx, result):
    if result.selection is not None:
        self._write_selection(ctx, result.selection, result.scores)

    if result.mapping is not None:
        self._write_mapping(ctx, result.mapping)

    if result.scores is not None:
        self._write_score_tensor(ctx, result.scores)
```

Selection 部分：

```python
def _write_selection(self, ctx, selection, scores=None):
    selected = selection.keep_global_indices.detach().cpu()
    batch_size = selected.shape[0]

    score_cpu = None
    if scores is not None:
        score_cpu = scores.detach().cpu()

    candidate = ctx.domain.candidate_indices.detach().cpu().tolist()
    selected_sets = [set(row.tolist()) for row in selected]

    for b in range(batch_size):
        for local_idx, global_idx in enumerate(candidate):
            score = None
            if score_cpu is not None:
                score = float(score_cpu[b, local_idx])

            frame, y, x = decode_video_index_checked(
                global_idx,
                ctx.layout,
            )

            self.token_rows.append({
                "scene_token": ctx.scene_token,
                "batch_index": b,
                "layer": ctx.layer_idx,
                "diffusion_rank": ctx.diffusion_rank,
                "original_index": global_idx,
                "latent_frame": frame,
                "spatial_y": y,
                "spatial_x": x,
                "score": score,
                "selected": global_idx in selected_sets[b],
            })
```

## 3.4 验收测试

```python
def test_noop_artifact_writer():
    result = NoPress().apply(ctx)
    writer.add_result(ctx, result)
    writer.write_tokens()

    assert output_exists()
    assert no_exception()
```

必须满足：

```text
NoPress + Evaluator + ArtifactWriter = PASS
```

---

# 4. P0-2：DriveVA Adapter 必须以 `press.domain` 为唯一默认真值来源

## 4.1 当前问题

文件：

```text
videopress/adapters/driveva.py
```

当前 hook 创建 context 时使用：

```python
runtime.current_sample.metadata.get("domain", "last_history")
```

因此即使配置：

```yaml
press:
  domain:
    name: history
```

Adapter 仍可能实际运行：

```text
last_history
```

这属于最危险的一类问题：

```text
程序正常执行
但实验条件错误
```

## 4.2 正确逻辑

默认必须：

```python
domain_spec = getattr(runtime.press, "domain", None)

if domain_spec is None:
    domain_spec = "last_history"
```

然后：

```python
context = self.create_context(
    ...,
    domain=domain_spec,
)
```

如果未来需要 sample override，应显式配置：

```yaml
runtime:
  allow_sample_domain_override: false
```

只有启用后才：

```python
if runtime.allow_sample_domain_override:
    domain_spec = sample.metadata.get("domain", domain_spec)
```

## 4.3 强制 metadata 记录

每个 result 写入：

```json
{
  "configured_domain": "history",
  "resolved_domain": "history",
  "candidate_start": 0,
  "candidate_end": 780,
  "n_candidate": 780
}
```

然后：

```python
assert configured_domain == resolved_domain
```

除非显式 override。

---

# 5. P0-3：统一 Gradient Objective 接口，禁止 silent objective degradation

## 5.1 当前问题

当前 `GradientScorer` 调用：

```python
objective.compute(outputs, ctx)
```

但 objective API 命名仍然是：

```python
def compute(self, outputs, batch=None):
```

且 `TrajectoryObjective` 通过：

```python
batch.get(target_key)
```

或者：

```python
getattr(batch, target_key, None)
```

查 target。

Evaluator 中真正存入的是：

```python
ctx.metadata["target_trajectory"] = sample.target_trajectory
```

这使 objective 数据路径含糊，并且 target 缺失时会退化成：

```python
trajectory.float().pow(2).mean()
```

对 planning attribution 来说，这不是安全 fallback。

## 5.2 统一 API

```python
class PlanningObjective:
    def compute(
        self,
        outputs,
        ctx: TokenContext,
    ) -> torch.Tensor:
        raise NotImplementedError
```

Trajectory：

```python
class TrajectoryObjective(PlanningObjective):
    def compute(self, outputs, ctx):
        trajectory = get_trajectory(outputs)

        target = ctx.metadata.get("target_trajectory")

        if target is None:
            raise RuntimeError(
                "TrajectoryObjective requires "
                "ctx.metadata['target_trajectory']"
            )

        target = target.to(
            device=trajectory.device,
            dtype=trajectory.dtype,
        )

        if target.shape != trajectory.shape:
            raise RuntimeError(
                f"trajectory shape mismatch: "
                f"pred={trajectory.shape}, target={target.shape}"
            )

        return (
            trajectory - target
        ).float().pow(2).mean()
```

Endpoint：

```python
class EndpointObjective(PlanningObjective):
    def compute(self, outputs, ctx):
        trajectory = get_trajectory(outputs)
        target = require_target(ctx)

        pred_endpoint = trajectory[:, -1]
        target_endpoint = target[:, -1].to(pred_endpoint.device)

        return (
            pred_endpoint - target_endpoint
        ).float().pow(2).mean()
```

## 5.3 禁止自动 target fallback

正式 experiment 下：

```text
target missing -> FAIL
```

如果确实需要无 target objective，应该建立独立类：

```python
TrajectoryMagnitudeObjective
```

不要让 `TrajectoryObjective` 自动改变语义。

---

# 6. P0-4：删除 `_call_with_context()` 的 TypeError fallback

## 6.1 当前问题

```python
def _call_with_context(fn, value, ctx):
    try:
        return fn(value, ctx)
    except TypeError:
        return fn(value)
```

若 `fn(value, ctx)` 内部真正产生 TypeError，框架会错误地解释为“函数不接受 ctx”。

## 6.2 修复

统一要求：

```python
forward_fn(tokens, ctx)
objective.compute(outputs, ctx)
```

旧接口使用显式 adapter：

```python
def adapt_legacy_forward(fn):
    def wrapped(tokens, ctx):
        return fn(tokens)
    return wrapped
```

Gradient 中直接：

```python
outputs = forward_fn(x, ctx)
loss = objective.compute(outputs, ctx)
```

任何 `TypeError` 原样向上抛出。

---

# 7. P0-5：真正实现 `VIDEO_INPUT` causal injection

## 7.1 当前状态

当前 `DriveVAAdapter.install_hooks()` 实际只 wrap：

```text
SelfAttention.attn.forward
```

即 post-RoPE：

```text
Q/K/V
  ↓
VideoPress hook
  ↓
attention kernel
```

这可以支持 `SELF_ATTN_KV`，但不能实现：

```yaml
injection_point: video_input
operator: zero
```

因此 Attention/Gradient/Random causal masking 仍无法通过真实 DriveVA pipeline 统一复现。

## 7.2 推荐插入位置

DriveVA pipeline：

```text
VAE latent
   ↓
patchify
   ↓
flatten video tokens
   ↓
[VIDEO_INPUT PRESS]
   ↓
concat trajectory tokens
   ↓
DiT blocks
```

不要在 concat trajectory 后再用 video-only operator，以避免 token layout 语义混淆。

## 7.3 推荐接口

在 pipeline flatten 后加入一个最小 hook：

```python
video_tokens = rearrange(
    x,
    "b c f h w -> b (f h w) c",
).contiguous()

if tokenpress_runtime is not None:
    video_tokens = tokenpress_runtime.apply_video_input(
        video_tokens=video_tokens,
        f=f,
        h=h,
        w=w,
        num_cond_latents=num_cond_latents,
        traj_len=traj_len,
        traj_prefix_len=traj_prefix_len,
    )
```

Runtime：

```python
def apply_video_input(...):
    if self.press.injection_point != InjectionPoint.VIDEO_INPUT:
        return video_tokens

    layout = self.adapter.build_layout(...)
    self.set_layout(layout)

    domain = resolve_press_domain(self.press, layout, video_tokens.device)

    ctx = TokenContext(
        tokens=video_tokens,
        layout=layout,
        domain=domain,
        ...
    )

    result = self.execute_press(ctx)
    self.record_result(ctx, result)

    if result.output.shape != video_tokens.shape:
        raise RuntimeError(
            "VIDEO_INPUT V1 only supports length-preserving causal operators"
        )

    return result.output
```

## 7.4 V1 限制

`VIDEO_INPUT` 第一版只允许：

```text
ZeroMask
MeanReplace
selected/drop shuffle（修复后）
```

禁止：

```text
Drop
Merge
physical prune
```

原因是 DriveVA 后续需要完整 video latent topology。

---

# 8. P0-6：Adapter 必须严格按 InjectionPoint 路由

## 8.1 当前问题

Adapter 当前无条件安装 attention hook。

这导致 `InjectionPoint` 只是配置 metadata，而不是执行协议。

## 8.2 修复

```python
def install_hooks(self, pipe, runtime):
    if runtime.press is None:
        return

    point = runtime.press.injection_point

    if point == InjectionPoint.VIDEO_INPUT:
        self._install_video_input_hook(pipe, runtime)
        return

    if point == InjectionPoint.SELF_ATTN_KV:
        self._install_kv_hooks(pipe, runtime)
        return

    if point == InjectionPoint.BLOCK_INPUT:
        raise NotImplementedError(
            "BLOCK_INPUT is declared but not implemented"
        )

    if point == InjectionPoint.SELF_ATTN_OUTPUT:
        raise NotImplementedError(
            "SELF_ATTN_OUTPUT is declared but not implemented"
        )

    raise RuntimeError(point)
```

核心原则：

```text
Unsupported != silently ignored
Unsupported -> fail fast
```

---

# 9. P0-7：修复 SimilarityMerge 在 DriveVA attention 中实际 no-op

## 9.1 当前问题

Adapter 只使用：

```python
result.aux["k"]
result.aux["v"]
```

若不存在则：

```python
return q, k, v
```

`KVPruneOperator` 会返回 K/V，因此能生效。

但当前 `MergeOperator` 只返回：

```text
output
mapping
metadata
```

没有 `aux['k'] / aux['v']`。

因此：

```text
SimilarityMergePress 内部看起来 length 下降
但真实 attention 仍收到完整 K/V
```

## 9.2 不要复用 hidden-token merge operator

拆成：

```text
HiddenTokenMergeOperator
KVMergeOperator
```

DriveVA physical merge 使用 `KVMergeOperator`。

## 9.3 K/V merge 伪代码

```python
class KVMergeOperator(TokenOperator):
    name = "kv_merge"

    def apply_plan(self, ctx, plan):
        if ctx.k is None or ctx.v is None:
            raise RuntimeError("KVMerge requires ctx.k and ctx.v")

        k_new = merge_qkv_sequence(ctx.k, plan)
        v_new = merge_qkv_sequence(ctx.v, plan)

        mapping = build_mapping(plan)

        return OperatorResult(
            output=ctx.tokens,
            mapping=mapping,
            metadata={
                "operator": self.name,
                "kv_length_before": ctx.k.shape[2],
                "kv_length_after": k_new.shape[2],
            },
            aux={
                "q": ctx.q,
                "k": k_new,
                "v": v_new,
            },
        )
```

Sequence merge：

```python
def merge_qkv_sequence(x, plan):
    # x: [B, H, L, Dh]
    outputs = []

    for group in plan.groups:
        idx = tensor(group.source_indices)
        values = x.index_select(2, idx)

        weights = tensor(group.weights)
        weights = weights / weights.sum()

        merged = (
            values
            * weights[None, None, :, None]
        ).sum(dim=2, keepdim=True)

        outputs.append(merged)

    return torch.cat(outputs, dim=2)
```

输出要求：

```text
Q length unchanged
K length reduced
V length reduced
```

即：

\[
L_Q=L,\qquad L_K=L_V=L'<L
\]

---

# 10. P0-8：修复 timing，区分 selector / model / e2e

## 10.1 当前问题

Evaluator 当前：

```python
timer.start()
result = press.apply(ctx)
timing = timer.stop()

prediction = predict_fn(...)
```

所以 `latency_ms` 只测：

```text
Scorer + Selector + Operator
```

不包含实际 DriveVA forward。

不能解释为：

```text
compressed model latency
```

## 10.2 正确输出至少三组时间

```text
selector_latency_ms
model_latency_ms
e2e_latency_ms
```

对于真实 DriveVA physical compression：

```python
e2e_timer.start()

with runtime.activate(pipe):
    output = pipe(...)

e2e = e2e_timer.stop()
```

如果需要 selector overhead：

```python
runtime.profiler.start("selector")
...
runtime.profiler.stop("selector")
```

## 10.3 显存也必须测真实 forward

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

with runtime.activate(pipe):
    output = pipe(...)

torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated()
```

不能仅测 `press.apply()`。

---

# 11. P0-9：接入 Official DriveVA + NAVSIM evaluator

## 11.1 当前状态

当前 `evaluation/navsim_evaluator.py` 主要承担 cohort/order 校验，并未形成：

```text
DriveVA inference
→ trajectory
→ official feature builder
→ NAVSIM metric cache
→ PDM
```

完整路径。

Synthetic backend 可以保留，但必须明确命名：

```text
backend=synthetic
```

不能将其结果与 official PDM 混淆。

## 11.2 推荐结构

```python
class DriveVANavsimBackend:
    def __init__(
        self,
        pipe,
        official_feature_builder,
        metric_cache_loader,
        adapter,
    ):
        ...

    def predict(self, sample, runtime):
        runtime.begin_sample(sample)

        with runtime.activate(self.pipe):
            output = run_official_driveva_inference(
                self.pipe,
                sample,
            )

        return extract_trajectory(output)

    def evaluate(self, trajectory, sample):
        return compute_official_navsim_pdm(
            trajectory=trajectory,
            scene=sample,
            feature_builder=self.feature_builder,
            metric_cache=self.metric_cache,
        )
```

## 11.3 Full baseline

Full 必须也走同一 backend：

```python
backend.evaluate(
    press=NoPress(),
)
```

不允许：

```text
Full → old evaluator
Compressed → new evaluator
```

---

# 12. P1-1：实现 automatic equal-budget Random baseline

## 12.1 当前问题

配置已有：

```yaml
evaluation:
  random_baseline: false
```

但 evaluator 未实际消费这一设置。

## 12.2 Random baseline 必须自动匹配

Method 与 Random 必须相同：

```text
Domain
Budget
Operator
InjectionPoint
Layer
Diffusion step/rank policy
Cohort
```

唯一变化：

```text
Scorer → RandomScorer
```

伪代码：

```python
def build_matched_random_press(method_press, seed):
    if not isinstance(method_press, ScorerPress):
        raise NotImplementedError

    return ScorerPress(
        scorer=RandomScorer(
            seed=seed,
            scope=method_press.random_scope,
        ),
        selector=deepcopy(method_press.selector),
        operator=deepcopy(method_press.operator),
        budget=deepcopy(method_press.budget),
        domain=deepcopy(method_press.domain),
        injection_point=method_press.injection_point,
    )
```

Evaluator：

```python
method = evaluate_once(press)

if config.evaluation.random_baseline:
    random_runs = []

    for seed in config.evaluation.random_seeds:
        random_press = build_matched_random_press(press, seed)
        random_runs.append(
            evaluate_once(random_press)
        )

    comparison = paired_compare(
        method,
        aggregate_random_by_scene(random_runs),
    )
```

---

# 13. P1-2：Random mask 不应依赖 batch_index

## 13.1 当前问题

当前 seed 包含：

```python
batch_index
```

因此同一 scene 只要 batch packing 改变，random mask 就改变。

## 13.2 修复

默认：

```python
seed = stable_seed(
    base_seed,
    scene_token,
    log_id,
    layer_key,
    diffusion_key,
)
```

不要使用：

```python
batch_index
```

如果同 scene 可能出现多个不同样本，使用稳定的：

```text
sample_uid / frame_token / timestamp
```

---

# 14. P1-3：为 Random 增加 scope

建议：

```yaml
scorer:
  name: random
  seed: 0
  scope: scene
```

支持：

```text
scene
scene_step
scene_layer
scene_layer_step
```

伪代码：

```python
def random_seed_for_ctx(self, ctx):
    parts = [self.seed, ctx.scene_token]

    if self.scope in {"scene_step", "scene_layer_step"}:
        parts.append(ctx.diffusion_rank)

    if self.scope in {"scene_layer", "scene_layer_step"}:
        parts.append(ctx.layer_idx)

    return stable_seed(*parts)
```

这样动态 selector 的 random control 才可严格匹配。

---

# 15. P1-4：ScoreCache 必须真正接入 probe → intervention

## 15.1 当前问题

Evaluator 虽然逻辑上分开：

```python
scores = press.score(ctx.clone_for_probe())
result = press.apply_with_scores(ctx, scores.detach())
```

但仍在同一个 evaluator 调用内使用内存 tensor，尚未形成真正的 frozen artifact protocol。

## 15.2 推荐 key

```python
ScoreKey(
    scene_token,
    diffusion_rank,
    layer_idx,
    scorer_signature,
)
```

## 15.3 Probe pass

```python
scores = press.score(probe_ctx)

cache.save(
    key,
    scores.detach().cpu(),
)

ranking = stable_argsort(scores)
cache.save_ranking(key, ranking)
```

记录：

```text
score digest
ranking digest
```

## 15.4 Intervention pass

必须 fresh forward：

```python
scores = cache.load(key, device)

result = press.apply_with_scores(
    intervention_ctx,
    scores,
)
```

绝不重新计算 ranking。

---

# 16. P1-5：修复 Attention × VNorm single-head 定义

## 16.1 当前问题

single head Attention：

```python
candidate_attn[:, head_index]
```

但 V norm 当前：

```python
norm(v_candidate).mean(dim=1)
```

等价于：

\[
A_{h^*}\times \frac1H\sum_h\|V_h\|
\]

而不是严格 matched-head：

\[
A_{h^*}\times\|V_{h^*}\|
\]

## 16.2 建议配置

```yaml
scorer:
  name: action_attention_vnorm
  head_mode: single
  head_index: 22
  value_norm_head_mode: matched
```

实现：

```python
if self.value_norm:
    v_candidate = ...

    if self.head_mode == "single" and self.value_norm_head_mode == "matched":
        v_norm = torch.linalg.vector_norm(
            v_candidate[:, self.head_index],
            dim=-1,
        )
    else:
        v_norm = torch.linalg.vector_norm(
            v_candidate,
            dim=-1,
        ).mean(dim=1)

    scores = scores * v_norm
```

Metadata 必须记录：

```json
{
  "head_mode": "single",
  "head_index": 22,
  "value_norm_head_mode": "matched"
}
```

---

# 17. P1-6：修复 ShuffleOperator 忽略 selection/K

## 17.1 当前问题

当前：

```python
candidate = ctx.domain.candidate_indices
values = x.index_select(1, candidate)
shuffle(all candidates)
```

因此：

```text
K=20%
K=80%
```

实际 intervention 完全一样。

## 17.2 不建议保留模糊 `shuffle`

拆成：

```text
shuffle_all
shuffle_drop
shuffle_keep
```

例如 `shuffle_drop`：

```python
class ShuffleDroppedOperator:
    def apply(self, ctx, selection):
        x = ctx.tokens.clone()

        for b in range(ctx.batch_size):
            drop_idx = selection.drop_global_indices[b]

            seed = stable_seed(
                self.seed,
                ctx.scene_token,
                "shuffle_drop",
            )

            perm = randperm(len(drop_idx), seed)

            shuffled = x[b, drop_idx[perm]]
            x[b, drop_idx] = shuffled

        return OperatorResult(...)
```

如果需要“整个 domain shuffle”作为独立 causal baseline，则实现：

```text
ShuffleAllPress
```

不要伪装成受 K 控制的 ScorerPress operator。

---

# 18. P1-7：Runtime 改为事件式记录

## 18.1 当前问题

现在：

```python
runtime.last_result = result
```

DriveVA 真实 forward 有：

```text
多个 diffusion step
×
多个 transformer layer
```

最后只有一个 result 可见。

## 18.2 推荐数据结构

```python
@dataclass(frozen=True)
class CompressionEventKey:
    scene_token: str
    diffusion_rank: int | None
    layer_idx: int | None
    injection_point: str


@dataclass
class CompressionEvent:
    key: CompressionEventKey
    result: CompressionResult
```

Runtime：

```python
self.events: list[CompressionEvent] = []


def record_result(self, ctx, result):
    self.events.append(
        CompressionEvent(
            key=CompressionEventKey(
                scene_token=ctx.scene_token,
                diffusion_rank=ctx.diffusion_rank,
                layer_idx=ctx.layer_idx,
                injection_point=self.press.injection_point.value,
            ),
            result=result,
        )
    )

    self.last_result = result  # debug only
```

每个 sample：

```python
def begin_sample(...):
    ...
    self.last_result = None
    self.events = []
    self.artifacts = {}
```

---

# 19. P1-8：EvaluationMode 必须成为真正的 protocol validator

## 19.1 当前问题

现在可以配置：

```yaml
mode: causal
operator: kv_prune
```

或者：

```yaml
mode: physical
operator: zero
```

框架未必拒绝。

## 19.2 Operator capability

```python
class TokenOperator:
    preserves_sequence_length = True
    physical_compression = False


class ZeroMaskOperator(TokenOperator):
    preserves_sequence_length = True
    physical_compression = False


class KVPruneOperator(TokenOperator):
    preserves_sequence_length = False
    physical_compression = True
```

## 19.3 Protocol validator

```python
def validate_protocol(press, mode):
    operator = getattr(press, "operator", None)
    point = press.injection_point

    if mode == EvaluationMode.CAUSAL:
        if operator is not None and operator.physical_compression:
            raise RuntimeError(
                "causal mode cannot use physical compression operator"
            )

    if mode == EvaluationMode.PHYSICAL:
        if operator is not None and not operator.physical_compression:
            raise RuntimeError(
                "physical mode requires real sequence/compute compression"
            )

    if point == InjectionPoint.VIDEO_INPUT:
        if operator is not None and not operator.preserves_sequence_length:
            raise RuntimeError(
                "VIDEO_INPUT V1 requires sequence-length preserving operator"
            )
```

---

# 20. P2-1：拆分 selection / score / mapping artifact

当前 artifact API 应从：

```python
add_tokens(ctx, result)
```

重构为：

```python
add_result(ctx, result)
```

内部：

```text
selection -> masks/
scores    -> scores/
mapping   -> mappings/
tokens    -> token table
```

推荐文件命名包含：

```text
scene
layer
diffusion rank
scorer signature
```

例如：

```text
artifacts/
  scores/
    scene123_rank0_layer15_action_attention.pt
  masks/
    scene123_rank0_layer15_k156.pt
  mappings/
    scene123_rank0_layer15_merge.json
```

否则不同 layer/step 会覆盖同一 `{scene_token}.json`。

---

# 21. P2-2：SimilarityMerge 明确 batch policy

## 21.1 当前问题

当前 feature：

```python
ctx.candidate_tokens().mean(dim=0)
```

即把 batch 平均成一个共同 merge plan。

## 21.2 V1 推荐直接限制 batch=1

```python
if ctx.batch_size != 1:
    raise NotImplementedError(
        "SimilarityMergePress V1 supports batch_size=1 only"
    )
```

这是最安全的方案。

后续再实现 batch-specific：

```text
MergePlan[B]
```

而不是 silent batch averaging。

---

# 22. P2-3：Video index decode 加范围检查

当前 decode 应升级为：

```python
def decode_video_index_checked(index, layout):
    if index < layout.video.start or index >= layout.video.end:
        raise ValueError(
            f"token {index} is not a video token; "
            f"video range={layout.video}"
        )

    local = index - layout.video.start

    frame = local // (layout.video_h * layout.video_w)
    rem = local % (layout.video_h * layout.video_w)
    y = rem // layout.video_w
    x = rem % layout.video_w

    return frame, y, x
```

避免 trajectory token 被错误映射为 `(f,y,x)`。

---

# 23. 推荐修复后的真实执行架构

```text
                         DriveVA
                            │
                     VAE / patchify
                            │
                     video tokens
                            │
                ┌───────────▼───────────┐
                │ VIDEO_INPUT hook      │
                │ causal only           │
                │ Zero/Mean/...         │
                └───────────┬───────────┘
                            │
                    concat trajectory
                            │
                     Transformer
                            │
                 Q/K/V norm + RoPE
                            │
                ┌───────────▼───────────┐
                │ SELF_ATTN_KV hook     │
                │ physical only         │
                │ KVPrune / KVMerge     │
                └───────────┬───────────┘
                            │
                       Attention
                            │
                       trajectory
                            │
                 Official NAVSIM PDM
```

两个 benchmark protocol 严格分开：

```text
CAUSAL:
    full sequence length
    information intervention
    no acceleration claim

PHYSICAL:
    K/V sequence truly reduced
    quality + latency + memory
```

---

# 24. 修复后 Evaluator 推荐伪代码

```python
def evaluate_experiment(config):
    press = build_press(config.press)
    validate_protocol(press, config.evaluation.mode)

    backend = build_backend(config.benchmark)

    method_records = evaluate_once(
        backend=backend,
        press=press,
        config=config,
    )

    comparison = None

    if config.evaluation.random_baseline:
        random_runs = []

        for seed in config.evaluation.random_seeds:
            random_press = build_matched_random_press(
                press,
                seed,
            )

            random_runs.append(
                evaluate_once(
                    backend=backend,
                    press=random_press,
                    config=config,
                )
            )

        comparison = paired_method_random_analysis(
            method_records,
            random_runs,
        )

    write_summary(
        method_records,
        comparison,
    )
```

`evaluate_once()`：

```python
def evaluate_once(backend, press, config):
    records = []

    for sample in backend.iter_samples():
        runtime = VideoPressRuntime(
            press=press,
            mode=config.evaluation.mode,
            adapter=DriveVAAdapter(),
        )

        runtime.begin_sample(sample)

        # Offline attribution method
        if press.requires_probe:
            run_probe_and_cache(
                backend,
                runtime,
                sample,
            )

        timer.start_e2e()

        prediction = backend.predict(
            sample,
            runtime,
        )

        timing = timer.stop_e2e()

        metrics = backend.evaluate(
            prediction,
            sample,
        )

        records.append(
            build_record(
                sample,
                runtime,
                metrics,
                timing,
            )
        )

        artifact_writer.add_runtime_events(runtime)

    return records
```

---

# 25. 修复后的测试分层

## Level 1：纯单元测试

必须覆盖：

```text
layout
last_history domain
history domain
budget
stable top-k
random determinism
operator protected invariant
KV gather shape
merge mapping
protocol validator
```

---

## Level 2：Framework integration

### Test A：NoPress

```python
result = Evaluator(...).evaluate(
    press=NoPress(),
)

assert result.valid
assert artifact_complete
```

### Test B：Random Zero

```text
same scene + same seed + different batch packing
=> same mask
```

### Test C：Exact-K

```text
configured K = 156
actual selected = 156
```

### Test D：Protected invariant

```text
non-candidate max abs diff = 0
```

---

## Level 3：DriveVA integration

### Gate 1：NoPress equivalence

```text
Official DriveVA
vs
Framework + NoPress
```

要求：

```text
trajectory identical / numerical tolerance
PDM identical
```

### Gate 2：VIDEO_INPUT causal

```text
Random Top-K + Zero
```

要求：

```text
candidate only modified
protected unchanged
full sequence length unchanged
```

### Gate 3：Attention causal

```text
Attention ranking
→ frozen score/ranking
→ fresh intervention pass
```

### Gate 4：Gradient causal

```text
Gradient objective target verified
→ frozen score/ranking
→ fresh intervention pass
```

### Gate 5：Physical KV prune

要求：

```text
LQ == full L
LK < full L
LV < full L
finite trajectory
valid PDM
```

### Gate 6：Physical KV merge

要求：

```text
LK/LV actually reduced
mapping complete
attention receives compressed K/V
```

---

# 26. 对旧 DriveVA 实验的回归测试

新框架正式启用前，至少复现四个旧条件：

```text
1. Full
2. Random Top-K Only / ZeroMask
3. Attention Top-K Only / ZeroMask
4. Gradient×Input Top-K Only / ZeroMask
```

同样：

```text
cohort
scene windows
feature builder
K definition
candidate domain
random seed
objective
```

都必须冻结。

目标不是得到“相似趋势”，而是验证：

```text
新框架不会改变已有实验定义
```

建议设：

```text
PDM absolute diff <= 1e-6
```

若 metric pipeline 存在数值非确定性，则先确定官方可接受容差再登记。

---

# 27. 推荐新增测试文件

```text
tests/
├── test_artifacts.py
├── test_protocol.py
├── test_objectives.py
├── test_random_scope.py
├── test_runtime_events.py
├── test_video_input_hook.py
├── test_kv_merge.py
├── test_noop_official_equivalence.py
└── test_navsim_backend.py
```

关键测试：

```python
def test_trajectory_objective_requires_target():
    ctx.metadata.pop("target_trajectory", None)

    with pytest.raises(RuntimeError):
        objective.compute(outputs, ctx)
```

```python
def test_random_independent_of_batch_position():
    mask_a = score_scene_in_batch_position(scene, pos=0)
    mask_b = score_scene_in_batch_position(scene, pos=3)

    assert torch.equal(mask_a, mask_b)
```

```python
def test_similarity_merge_changes_real_kv_length():
    q2, k2, v2 = run_driveva_kv_hook(...)

    assert q2.shape[2] == q.shape[2]
    assert k2.shape[2] < k.shape[2]
    assert v2.shape[2] < v.shape[2]
```

---

# 28. 修复实施顺序

严格建议按以下顺序执行：

```text
Step 1
ArtifactWriter：NoPress + mapping 修复

Step 2
Gradient Objective API + 删除 TypeError fallback

Step 3
press.domain resolution 修复

Step 4
InjectionPoint protocol validator

Step 5
DriveVA VIDEO_INPUT causal hook

Step 6
NoPress official DriveVA equivalence

Step 7
Random / Attention / Gradient causal 回归

Step 8
ScoreCache + two-pass frozen ranking

Step 9
Random matched baseline + random scope

Step 10
Official NAVSIM backend

Step 11
真实 e2e latency / memory profiler

Step 12
KV prune official integration

Step 13
KVMergeOperator + SimilarityMerge 修复

Step 14
Runtime event/artifact system

Step 15
batch-aware merge / 其他扩展
```

不要在 Step 6 之前继续接入新的论文方法。

---

# 29. 最终验收 Checklist

## Framework correctness

- [ ] `python -m compileall` PASS
- [ ] Unit tests 全绿
- [ ] DriveVA integration test 不因独立 package 缺 `diffsynth` 而误报失败
- [ ] NoPress Evaluator 可运行
- [ ] NoPress artifact 可生成
- [ ] Merge mapping artifact 可生成
- [ ] unsupported injection point fail-fast

## Domain / Budget

- [ ] Adapter 使用 `press.domain`
- [ ] `configured_domain == resolved_domain`
- [ ] candidate range 正确
- [ ] exact-K 正确
- [ ] eligible/global keep ratio 同时记录
- [ ] protected token unchanged

## Gradient

- [ ] objective API 统一为 `(outputs, ctx)`
- [ ] target 缺失直接失败
- [ ] 无 silent objective fallback
- [ ] 无 TypeError signature fallback
- [ ] probe 与 intervention 为 fresh pass
- [ ] score/ranking digest 持久化

## Attention

- [ ] post-norm/post-RoPE Q/K
- [ ] action query range 正确
- [ ] softmax domain 明确
- [ ] single head VNorm 与 head 定义一致
- [ ] scorer metadata 完整

## Random

- [ ] same sample / seed 可复现
- [ ] batch packing 不改变 mask
- [ ] random scope 明确
- [ ] equal-budget Random 自动生成

## Causal path

- [ ] `VIDEO_INPUT` hook 真实接入
- [ ] sequence length unchanged
- [ ] Random Zero 可运行
- [ ] Attention Zero 可运行
- [ ] Gradient Zero 可运行
- [ ] 与旧实验结果回归一致

## Physical path

- [ ] post-RoPE K/V hook 正确
- [ ] KVPrune 实际减少 K/V length
- [ ] Q length 不变
- [ ] KVMerge 实际减少 K/V length
- [ ] physical mapping 完整
- [ ] trajectory/PDM 有效

## NAVSIM

- [ ] official feature builder
- [ ] no silent fallback
- [ ] safe scene windows
- [ ] cross-scene=0
- [ ] Full 与 compressed 走同一 evaluator
- [ ] official PDM 输出

## Efficiency

- [ ] warmup
- [ ] CUDA synchronize
- [ ] selector latency
- [ ] model latency
- [ ] e2e latency
- [ ] peak model memory
- [ ] theoretical K/V attention ratio
- [ ] ZeroMask 不报告为 physical speedup

## Artifacts

- [ ] config snapshot
- [ ] environment snapshot
- [ ] git commit
- [ ] checkpoint
- [ ] seed
- [ ] scene records
- [ ] token score/mask
- [ ] layer/diffusion identifiers
- [ ] score/ranking digest
- [ ] merge/prune mapping
- [ ] no cross-layer artifact overwrite

---

# 30. 修复后的最低正式可用版本定义

建议定义：

```text
VideoTokenPress v0.1 benchmark-ready
```

只有同时满足以下 6 个 Gate 才可以打这个标签：

### Gate A — Full equivalence

```text
Framework NoPress == Official DriveVA
```

### Gate B — Causal intervention

```text
Random / Attention / Gradient
可以统一通过 VIDEO_INPUT intervention
```

### Gate C — Frozen ranking

```text
Probe score 与 intervention forward 严格分离
```

### Gate D — Official metric

```text
所有实验通过同一 official NAVSIM PDM evaluator
```

### Gate E — Equal-budget control

```text
所有 selector 默认可生成严格匹配的 Random baseline
```

### Gate F — Physical compression

```text
至少 Random-KV-Prune 能真实降低 K/V sequence length，
并获得有效 PDM + latency + memory
```

Merge 不必阻塞 v0.1，但在修复前必须明确标记：

```text
SimilarityMerge: EXPERIMENTAL / DISABLED
```

而不能作为可用 physical benchmark method。

---

# 31. 最终判断

当前代码不需要重新设计核心架构。建议保留：

```text
TokenLayout
TokenDomain
TokenBudget
TokenContext
Scorer
Selector
Operator
BaseVideoPress / ScorerPress
DriveVAAdapter
```

当前最主要的工程任务是把：

```text
“独立 token manipulation SDK”
```

真正闭环成：

```text
Official DriveVA Input
      ↓
Resolved Token Domain
      ↓
VideoTokenPress
      ↓
Causal VIDEO_INPUT
or Physical SELF_ATTN_KV
      ↓
Official DriveVA Forward
      ↓
Trajectory
      ↓
Official NAVSIM PDM
      ↓
Matched Random + Statistics
      ↓
Latency / Memory / Artifacts
```

在这一闭环完成前，不建议继续大规模加入新的 Attention、ToMe、TokenLearner、Resampler 等方法。否则方法数量增加，但 benchmark protocol 本身仍可能改变实验语义。

当前最优先的验收目标应压缩为四个可以一键运行且可与旧实验对齐的条件：

```text
Full / NoPress
Random Top-K Zero
Attention Top-K Zero
Random KV Prune
```

这四个条件正式通过 DriveVA + official NAVSIM 后，再扩展 Gradient、Merge 和其他论文方法。
