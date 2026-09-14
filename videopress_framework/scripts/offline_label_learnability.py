#!/usr/bin/env python3
"""Offline supervised upper bound for the tile-level counterfactual teacher.

Review 2026-09-11 P0-2.  The online selector reached held-out AUC ~0.50, but that
result conflates two very different worlds:

* the labels carry no information about the features the selector sees, or
* the labels do carry information that the online loss/optimiser fails to find.

This script settles it by fitting strong supervised models *directly* on the
dumped probes (features the selector saw, labels the teacher produced) and
reporting held-out AUCs with bootstrap intervals and permutation p-values.

A positive control re-runs the identical pipeline on a label synthesised from the
features; if the control cannot recover it, the harness is broken and no negative
verdict may be drawn from the real labels.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_probes(dirs: list[Path], max_probes: int | None = None) -> list[dict]:
    paths: list[str] = []
    for directory in dirs:
        paths.extend(sorted(glob.glob(str(directory / "rank*" / "step-*.pt"))))
    paths.sort()
    if max_probes is not None:
        paths = paths[: int(max_probes)]
    probes = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        probes.append(
            {
                "path": path,
                "sample_token": payload["sample_token"],
                "global_step": int(payload["global_step"]),
                "rank": int(payload["rank"]),
                "tokens": payload["tokens"][0].float().numpy(),
                "positions": payload["positions"][0].float().numpy(),
                "ego_state": payload["ego_state"][0].float().numpy(),
                "command": payload["command"][0].float().numpy(),
                "timestep": float(payload["selector_timestep"].reshape(-1)[0]),
                "selector_logits": payload["selector_logits"][0].float().numpy(),
                "membership": payload["membership"][0].numpy().astype(bool),
                "group_index": int(payload["group_index"]),
                "relative_delta": float(payload["relative_delta"]),
                "helpful_target": float(payload["helpful_target"]),
                "confidence": float(payload["confidence"]),
                "baseline_loss_unweighted": float(payload["baseline_loss_unweighted"]),
                "masked_loss_unweighted": float(payload["masked_loss_unweighted"]),
                # Optional: present only for runs that also recorded the
                # geometric plan displacement (review P3 item 3).
                "traj_disp": (
                    float(payload["counterfactual_traj_disp_mean"])
                    if payload.get("counterfactual_traj_disp_mean") is not None
                    else None
                ),
                "traj_disp_relative": (
                    float(payload["counterfactual_traj_disp_relative"])
                    if payload.get("counterfactual_traj_disp_relative") is not None
                    else None
                ),
            }
        )
    return probes


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n_pos = float((labels >= 0.5).sum())
    n_neg = float((labels < 0.5).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranked = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    index = 0
    rank = np.empty(len(scores), dtype=np.float64)
    while index < len(scores):
        end = index
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        rank[index : end + 1] = (index + end) / 2.0 + 1.0
        index = end + 1
    ranked[order] = rank
    u = ranked[labels >= 0.5].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def bootstrap_ci(scores: np.ndarray, labels: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    if n <= 0 or len(scores) < 8:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = []
    size = len(scores)
    for _ in range(n):
        take = rng.integers(0, size, size)
        value = auc(scores[take], labels[take])
        if value == value:
            draws.append(value)
    if len(draws) < 10:
        return float("nan"), float("nan")
    draws.sort()
    return (
        float(draws[int(0.025 * len(draws))]),
        float(draws[min(len(draws) - 1, int(0.975 * len(draws)))]),
    )


def permutation_p(scores: np.ndarray, labels: np.ndarray, n: int, seed: int) -> float:
    observed = auc(scores, labels)
    if observed != observed or n <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    extreme = 0
    work = labels.copy()
    for _ in range(n):
        rng.shuffle(work)
        value = auc(scores, work)
        if value == value and abs(value - 0.5) >= abs(observed - 0.5) - 1e-12:
            extreme += 1
    return float((extreme + 1) / (n + 1))


def evaluate(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> dict:
    value = auc(scores, labels)
    low, high = bootstrap_ci(scores, labels, bootstrap, seed)
    p_value = permutation_p(scores, labels, permutations, seed + 1)
    return {
        "name": name,
        "n": int(len(labels)),
        "positive_rate": float(np.mean(labels >= 0.5)) if len(labels) else None,
        "auc": value,
        "auc_ci95": [low, high],
        "permutation_p_value": p_value,
        "ci_excludes_chance": bool(low == low and high == high and (low > 0.5 or high < 0.5)),
    }


def scene_split(tokens: list[str], seed: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted({t for t in tokens})
    rng = random.Random(seed)
    rng.shuffle(groups)
    cut = max(1, int(len(groups) * train_fraction))
    train_groups = set(groups[:cut])
    train_idx = np.array([i for i, t in enumerate(tokens) if t in train_groups])
    eval_idx = np.array([i for i, t in enumerate(tokens) if t not in train_groups])
    return train_idx, eval_idx


def within_scene_ranking(
    scores: np.ndarray, deltas: np.ndarray, scenes: list[str]
) -> dict:
    """Pairwise concordance between a score and the measured delta, within a scene.

    This is the metric token selection is actually judged by: it does not reward
    a model for knowing that scene A is generally more sensitive than scene B, it
    only rewards ordering the tiles *of the same scene* correctly.
    """
    from collections import defaultdict

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for scene, delta, score in zip(scenes, deltas, scores):
        grouped[scene].append((float(delta), float(score)))

    concordant = total = 0
    rank_correlations = []
    scenes_used = 0
    for items in grouped.values():
        if len(items) < 2:
            continue
        scenes_used += 1
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                delta_i, score_i = items[i]
                delta_j, score_j = items[j]
                if delta_i == delta_j:
                    continue
                total += 1
                concordant += int((score_i - score_j) * (delta_i - delta_j) > 0)
        try:
            from scipy.stats import spearmanr

            deltas_only = [item[0] for item in items]
            scores_only = [item[1] for item in items]
            if len(set(scores_only)) > 1 and len(set(deltas_only)) > 1:
                rho, _ = spearmanr(deltas_only, scores_only)
                if rho == rho:
                    rank_correlations.append(float(rho))
        except ImportError:
            pass
    return {
        "pairwise_accuracy": float(concordant / total) if total else float("nan"),
        "n_pairs": int(total),
        "n_scenes": int(scenes_used),
        "mean_within_scene_spearman": (
            float(np.mean(rank_correlations)) if rank_correlations else float("nan")
        ),
        "n_spearman_scenes": len(rank_correlations),
    }


def fit_logistic(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int):
    if len(np.unique(y_train)) < 2:
        return np.full(len(x_eval), 0.5)
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=3000, C=0.05, solver="lbfgs", random_state=seed)
    model.fit(scaler.transform(x_train), y_train)
    return model.predict_proba(scaler.transform(x_eval))[:, 1]


def fit_gbdt(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int):
    if len(np.unique(y_train)) < 2:
        return np.full(len(x_eval), 0.5)
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=seed
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_eval)[:, 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_dirs", nargs="+", type=Path)
    parser.add_argument("--eval-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-probes", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--max-tokens-per-probe", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--skip-token-level", action="store_true")
    args = parser.parse_args()

    probes = load_probes(list(args.probe_dirs), args.max_probes)
    if not probes:
        raise SystemExit("no probes loaded")
    eval_probes = load_probes(list(args.eval_dirs), args.max_probes) if args.eval_dirs else []

    token_dim = probes[0]["tokens"].shape[1]
    n_groups = 12

    def build_features(items: list[dict]) -> dict:
        tile_mean = np.stack(
            [item["tokens"][item["membership"]].mean(axis=0) for item in items]
        )
        tile_std = np.stack(
            [item["tokens"][item["membership"]].std(axis=0) for item in items]
        )
        scene_mean = np.stack([item["tokens"].mean(axis=0) for item in items])
        pos_mean = np.stack(
            [item["positions"][item["membership"]].mean(axis=0) for item in items]
        )
        context = np.concatenate(
            [
                pos_mean,
                np.stack([item["ego_state"] for item in items]),
                np.stack([item["command"] for item in items]),
                np.array([[item["timestep"] / 1000.0] for item in items]),
            ],
            axis=1,
        )
        group_onehot = np.zeros((len(items), n_groups), dtype=np.float64)
        for row, item in enumerate(items):
            group_onehot[row, item["group_index"] % n_groups] = 1.0
        return {
            "tile_mean": tile_mean.astype(np.float64),
            "tile_std": tile_std.astype(np.float64),
            "scene_mean": scene_mean.astype(np.float64),
            "centered": (tile_mean - scene_mean).astype(np.float64),
            "context": np.concatenate([context, group_onehot], axis=1),
            "group_onehot": group_onehot,
            "raw_group": np.array([[item["group_index"]] for item in items], dtype=np.float64),
        }

    train_features = build_features(probes)
    helpful = np.array([item["helpful_target"] for item in probes])
    delta = np.array([item["relative_delta"] for item in probes])
    tokens = [item["sample_token"] for item in probes]
    logit_scores = np.array(
        [item["selector_logits"][item["membership"]].mean() for item in probes]
    )

    train_idx, eval_idx = scene_split(tokens, args.seed, args.train_fraction)
    report: dict = {
        "probe_dirs": [str(p) for p in args.probe_dirs],
        "eval_dirs": [str(p) for p in args.eval_dirs],
        "n_probes": len(probes),
        "n_train": int(len(train_idx)),
        "n_eval": int(len(eval_idx)),
        "token_dim": token_dim,
        "delta_stats": {
            "mean": float(delta.mean()),
            "std": float(delta.std()),
            "mean_abs": float(np.abs(delta).mean()),
            "helpful_rate": float((delta >= 0).mean()),
            "identity_zero_floor": None,
        },
        "results": {},
    }

    # ---- context-only: can the scene condition alone explain the label? -----
    x_ctx_train, x_ctx_eval = train_features["context"][train_idx], train_features["context"][eval_idx]
    y_train, y_eval = helpful[train_idx], helpful[eval_idx]
    report["results"]["context_only_logistic"] = evaluate(
        "context_only_logistic",
        fit_logistic(x_ctx_train, y_train, x_ctx_eval, args.seed),
        y_eval,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed,
    )
    report["results"]["context_only_gbdt"] = evaluate(
        "context_only_gbdt",
        fit_gbdt(x_ctx_train, y_train, x_ctx_eval, args.seed),
        y_eval,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed + 7,
    )
    report["results"]["online_selector_logits"] = evaluate(
        "online_selector_logits",
        logit_scores[eval_idx],
        y_eval,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed + 11,
    )
    report["results"]["group_index_only"] = evaluate(
        "group_index_only",
        logit_scores[eval_idx] * 0 + train_features["raw_group"][eval_idx, 0],
        y_eval,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed + 13,
    )

    # ---- token content: is the label predictable from the hidden states? ----
    n_components = min(args.pca_components, min(len(train_idx) - 1, token_dim))
    pca = PCA(n_components=n_components, random_state=args.seed)
    pca.fit(train_features["tile_mean"][train_idx])
    tm_train = pca.transform(train_features["tile_mean"][train_idx])
    tm_eval = pca.transform(train_features["tile_mean"][eval_idx])
    x_tok_train = np.concatenate([tm_train, x_ctx_train], axis=1)
    x_tok_eval = np.concatenate([tm_eval, x_ctx_eval], axis=1)
    report["pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())
    report["results"]["tile_mean_raw_logistic"] = evaluate(
        "tile_mean_raw_logistic",
        fit_logistic(
            train_features["tile_mean"][train_idx], y_train,
            train_features["tile_mean"][eval_idx], args.seed,
        ),
        y_eval,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed + 15,
    )
    report["results"]["tile_mean_pca_logistic"] = evaluate(
        "tile_mean_pca_logistic",
        fit_logistic(x_tok_train, y_train, x_tok_eval, args.seed + 1),
        y_eval,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed + 17,
    )
    report["results"]["tile_mean_pca_gbdt"] = evaluate(
        "tile_mean_pca_gbdt",
        fit_gbdt(x_tok_train, y_train, x_tok_eval, args.seed + 1),
        y_eval,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed + 19,
    )

    # ---- positive control ------------------------------------------------
    # The synthetic label is defined *inside the representation the model sees*
    # (PCA scores + context), so a working harness must recover it.  Defining it
    # in raw 3072-d space would make the control unlearnable for reasons that
    # have nothing to do with the label under test.
    rng = np.random.default_rng(args.seed)
    control_direction = rng.normal(size=x_tok_train.shape[1])
    control_train = (x_tok_train @ control_direction >= 0).astype(np.float64)
    control_eval = (x_tok_eval @ control_direction >= 0).astype(np.float64)
    report["results"]["POSITIVE_CONTROL_tile_mean_pca_logistic"] = evaluate(
        "POSITIVE_CONTROL_tile_mean_pca_logistic",
        fit_logistic(x_tok_train, control_train, x_tok_eval, args.seed + 2),
        control_eval,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed + 23,
    )
    # Second control in the *full* 3072-d space: it rules out the excuse that the
    # real label depends on directions outside the leading principal components.
    raw_direction = np.random.default_rng(args.seed + 202).normal(size=token_dim)
    raw_control_train = (
        train_features["tile_mean"][train_idx] @ raw_direction >= 0
    ).astype(np.float64)
    raw_control_eval = (
        train_features["tile_mean"][eval_idx] @ raw_direction >= 0
    ).astype(np.float64)
    report["results"]["POSITIVE_CONTROL_tile_mean_raw_logistic"] = evaluate(
        "POSITIVE_CONTROL_tile_mean_raw_logistic",
        fit_logistic(
            train_features["tile_mean"][train_idx], raw_control_train,
            train_features["tile_mean"][eval_idx], args.seed + 16,
        ),
        raw_control_eval,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed + 27,
    )
    # ---- negative control: shuffle the real labels -------------------------
    shuffled = helpful.copy()
    rng.shuffle(shuffled)
    report["results"]["NEGATIVE_CONTROL_shuffled_labels"] = evaluate(
        "NEGATIVE_CONTROL_shuffled_labels",
        fit_logistic(x_tok_train, shuffled[train_idx], x_tok_eval, args.seed + 3),
        shuffled[eval_idx],
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed + 29,
    )

    # ---- P1 dead-zone sweep, evaluated offline ---------------------------
    # ``helpful = delta >= 0`` turns a +-1e-4 measurement into a hard label.  If
    # even the *large* effects are unpredictable, the dead zone cannot rescue
    # the route; if only the large effects are predictable, it can.
    report["dead_zone_sweep"] = {}
    for eps in (0.0, 0.0005, 0.001, 0.002, 0.005):
        keep = np.abs(delta) > eps
        keep_train = keep[train_idx]
        keep_eval = keep[eval_idx]
        if keep_train.sum() < 20 or keep_eval.sum() < 20:
            report["dead_zone_sweep"][f"eps_{eps}"] = {
                "n_train": int(keep_train.sum()),
                "n_eval": int(keep_eval.sum()),
                "skipped": True,
            }
            continue
        label_dz = (delta >= 0).astype(np.float64)
        scores = fit_logistic(
            x_tok_train[keep_train], label_dz[train_idx][keep_train],
            x_tok_eval[keep_eval], args.seed + 12,
        )
        entry = evaluate(
            f"dead_zone_eps{eps}",
            scores,
            label_dz[eval_idx][keep_eval],
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            seed=args.seed + 61,
        )
        entry["n_train"] = int(keep_train.sum())
        entry["n_eval"] = int(keep_eval.sum())
        entry["abstain_ratio_train"] = float(1.0 - keep_train.mean())
        report["dead_zone_sweep"][f"eps_{eps}"] = entry

    # ---- continuous target: is the *magnitude* of the effect predictable? ---
    delta_threshold = float(np.median(np.abs(delta[train_idx])))
    extreme = (np.abs(delta) >= delta_threshold).astype(np.float64)
    report["results"]["extreme_delta_tile_mean_pca_logistic"] = evaluate(
        "extreme_delta_tile_mean_pca_logistic",
        fit_logistic(x_tok_train, extreme[train_idx], x_tok_eval, args.seed + 4),
        extreme[eval_idx],
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed + 31,
    )
    # Spearman correlation of the predicted probability with the signed delta.
    try:
        from scipy.stats import spearmanr

        delta_scores = fit_logistic(
            x_tok_train, (delta[train_idx] >= 0).astype(np.float64), x_tok_eval, args.seed + 14
        )
        if np.std(delta_scores) > 0 and np.std(delta[eval_idx]) > 0:
            rho, p_value = spearmanr(delta[eval_idx], delta_scores)
            report["results"]["signed_delta_spearman"] = {
                "name": "signed_delta_spearman",
                "n": int(len(eval_idx)),
                "rho": float(rho),
                "p_value": float(p_value),
            }
    except ImportError:
        report["results"]["signed_delta_spearman"] = {"name": "signed_delta_spearman", "skipped": True}

    # ---- token-level: can any per-token model rank tokens within a tile? ----
    if not args.skip_token_level:
        token_rows, token_labels, token_probe = [], [], []
        for index, item in enumerate(probes):
            selected = np.flatnonzero(item["membership"])[: args.max_tokens_per_probe]
            block = item["tokens"][selected]
            extra = np.tile(
                np.concatenate(
                    [item["positions"][selected], np.repeat(
                        np.concatenate([item["ego_state"], item["command"], [item["timestep"] / 1000.0]])[None, :],
                        len(selected),
                        axis=0,
                    )],
                    axis=1,
                ),
                (1, 1),
            )
            token_rows.append(np.concatenate([block, extra], axis=1))
            token_labels.append(np.full(len(selected), item["helpful_target"]))
            token_probe.append(np.full(len(selected), index))
        token_x = np.concatenate(token_rows, axis=0).astype(np.float64)
        token_y = np.concatenate(token_labels)
        token_p = np.concatenate(token_probe)

        train_mask = np.isin(token_p, train_idx)
        eval_mask = np.isin(token_p, eval_idx)
        token_pca = PCA(
            n_components=min(args.pca_components, token_x.shape[1]), random_state=args.seed
        ).fit(token_x[train_mask])
        tx_train = token_pca.transform(token_x[train_mask])
        tx_eval = token_pca.transform(token_x[eval_mask])
        report["token_level_n_train"] = int(train_mask.sum())
        report["token_level_n_eval"] = int(eval_mask.sum())
        report["results"]["token_level_pca_logistic"] = evaluate(
            "token_level_pca_logistic",
            fit_logistic(tx_train, token_y[train_mask], tx_eval, args.seed + 5),
            token_y[eval_mask],
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            seed=args.seed + 37,
        )
        # Positive control inside the token representation the model actually
        # sees, so "the harness works" is a real check rather than a restatement
        # of the dimensionality mismatch.
        token_control_direction = np.random.default_rng(args.seed + 101).normal(
            size=tx_train.shape[1]
        )
        token_control = (tx_train @ token_control_direction >= 0).astype(np.float64)
        token_control_eval = (tx_eval @ token_control_direction >= 0).astype(np.float64)
        report["results"]["POSITIVE_CONTROL_token_level"] = evaluate(
            "POSITIVE_CONTROL_token_level",
            fit_logistic(tx_train, token_control, tx_eval, args.seed + 6),
            token_control_eval,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            seed=args.seed + 41,
        )

    # ---- within-scene ranking: the metric token selection is judged by ------
    from sklearn.linear_model import Ridge

    ridge = Ridge(alpha=1.0, random_state=args.seed)
    ridge.fit(x_tok_train, delta[train_idx])
    eval_scenes = [tokens[i] for i in eval_idx]
    # Positive control for the ranking harness: a synthetic *continuous* target
    # that is an exact linear function of the features the model sees.  If the
    # harness cannot rank this, it cannot be trusted to rank the real delta.
    control_target = x_tok_train @ control_direction[: x_tok_train.shape[1]]
    control_ridge = Ridge(alpha=1.0, random_state=args.seed)
    control_ridge.fit(x_tok_train, control_target)
    report["within_scene_ranking"] = {
        "offline_ridge_on_measured_delta": within_scene_ranking(
            ridge.predict(x_tok_eval), delta[eval_idx], eval_scenes
        ),
        "online_selector_logits": within_scene_ranking(
            logit_scores[eval_idx], delta[eval_idx], eval_scenes
        ),
        "offline_logistic_on_sign": within_scene_ranking(
            fit_logistic(x_tok_train, y_train, x_tok_eval, args.seed + 21),
            delta[eval_idx],
            eval_scenes,
        ),
        "POSITIVE_CONTROL_offline_ridge": within_scene_ranking(
            control_ridge.predict(x_tok_eval),
            (x_tok_eval @ control_direction[: x_tok_eval.shape[1]]),
            eval_scenes,
        ),
        "note": (
            "pairwise_accuracy is over tile pairs of the same scene; 0.5 is chance. "
            "A scorer that only knows scene difficulty still scores 0.5 here, so this "
            "is strictly harder than the between-scene AUC reported above."
        ),
    }

    # ---- alternative target: geometric plan displacement (review P3) -------
    displacement = np.array(
        [np.nan if item["traj_disp"] is None else item["traj_disp"] for item in probes]
    )
    if not np.all(np.isnan(displacement)):
        report["displacement_target"] = {
            "n": int(np.sum(~np.isnan(displacement))),
            "mean": float(np.nanmean(displacement)),
            "std": float(np.nanstd(displacement)),
            "pearson_with_signed_delta": float(
                np.corrcoef(displacement[~np.isnan(displacement)], delta[~np.isnan(displacement)])[0, 1]
            ),
            "pearson_with_abs_delta": float(
                np.corrcoef(displacement[~np.isnan(displacement)], np.abs(delta)[~np.isnan(displacement)])[0, 1]
            ),
            "relative_dispersion": float(np.nanstd(displacement) / max(abs(np.nanmean(displacement)), 1e-9)),
            "loss_delta_relative_dispersion": float(np.std(delta) / max(abs(np.mean(delta)), 1e-9)),
        }
        disp_ridge = Ridge(alpha=1.0, random_state=args.seed)
        disp_ridge.fit(x_tok_train, displacement[train_idx])
        predicted_disp_eval = disp_ridge.predict(x_tok_eval)
        report["displacement_target"]["held_out_r2"] = float(
            1.0
            - np.sum((displacement[eval_idx] - predicted_disp_eval) ** 2)
            / max(np.sum((displacement[eval_idx] - displacement[train_idx].mean()) ** 2), 1e-12)
        )
        report["displacement_target"]["within_scene_ranking_by_measured_displacement"] = (
            within_scene_ranking(predicted_disp_eval, displacement[eval_idx], eval_scenes)
        )
        # The practical question: a model trained on displacement, then used to
        # rank tiles by measured usefulness.
        report["displacement_target"]["within_scene_ranking_by_predicted_disp_vs_delta"] = (
            within_scene_ranking(predicted_disp_eval, delta[eval_idx], eval_scenes)
        )

    # ---- option B: fit on the train dump, evaluate on a held-out dump -------
    if eval_probes:
        held_features = build_features(eval_probes)
        held_helpful = np.array([item["helpful_target"] for item in eval_probes])
        held_logits = np.array(
            [item["selector_logits"][item["membership"]].mean() for item in eval_probes]
        )
        held_pca = PCA(
            n_components=min(args.pca_components, min(len(probes) - 1, token_dim)),
            random_state=args.seed,
        ).fit(train_features["tile_mean"])
        x_fit = np.concatenate(
            [held_pca.transform(train_features["tile_mean"]), train_features["context"]], axis=1
        )
        x_held = np.concatenate(
            [held_pca.transform(held_features["tile_mean"]), held_features["context"]], axis=1
        )
        report["cross_dataset_n_eval"] = len(eval_probes)
        report["results"]["CROSS_DATASET_tile_mean_pca_logistic"] = evaluate(
            "CROSS_DATASET_tile_mean_pca_logistic",
            fit_logistic(x_fit, helpful, x_held, args.seed + 8),
            held_helpful,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            seed=args.seed + 43,
        )
        report["results"]["CROSS_DATASET_online_selector"] = evaluate(
            "CROSS_DATASET_online_selector",
            held_logits,
            held_helpful,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            seed=args.seed + 47,
        )
        cross_control_direction = np.random.default_rng(args.seed + 103).normal(
            size=x_fit.shape[1]
        )
        cross_control = (x_fit @ cross_control_direction >= 0).astype(np.float64)
        cross_control_held = (x_held @ cross_control_direction >= 0).astype(np.float64)
        report["results"]["CROSS_DATASET_POSITIVE_CONTROL"] = evaluate(
            "CROSS_DATASET_POSITIVE_CONTROL",
            fit_logistic(x_fit, cross_control, x_held, args.seed + 9),
            cross_control_held,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            seed=args.seed + 53,
        )
        # A dead-zone cross-dataset check: only large effects, signed.
        held_delta = np.array([item["relative_delta"] for item in eval_probes])
        for eps in (0.0, 0.002):
            keep_fit = np.abs(delta) > eps
            keep_held = np.abs(held_delta) > eps
            if keep_fit.sum() < 50 or keep_held.sum() < 20:
                continue
            entry = evaluate(
                f"CROSS_DATASET_dead_zone_eps{eps}",
                fit_logistic(
                    x_fit[keep_fit], (delta[keep_fit] >= 0).astype(np.float64),
                    x_held[keep_held], args.seed + 15,
                ),
                (held_delta[keep_held] >= 0).astype(np.float64),
                bootstrap=args.bootstrap,
                permutations=args.permutations,
                seed=args.seed + 59,
            )
            entry["n_fit"] = int(keep_fit.sum())
            entry["n_held"] = int(keep_held.sum())
            report["results"][f"CROSS_DATASET_dead_zone_eps{eps}"] = entry

    # ---- verdict -----------------------------------------------------------
    positive = report["results"].get("POSITIVE_CONTROL_tile_mean_pca_logistic", {})
    real = report["results"].get("tile_mean_pca_logistic", {})
    raw = report["results"].get("tile_mean_raw_logistic", {})
    context_only = report["results"].get("context_only_logistic", {})

    def _ok(value) -> bool:
        return value is not None and value == value

    control_auc = positive.get("auc")
    raw_control = report["results"].get("POSITIVE_CONTROL_tile_mean_raw_logistic", {})
    control_ok = (
        _ok(control_auc)
        and control_auc > 0.8
        and _ok(raw_control.get("auc"))
        and raw_control["auc"] > 0.8
    )
    real_auc = raw.get("auc") if _ok(raw.get("auc")) else real.get("auc")
    real_ci_ok = bool(raw.get("ci_excludes_chance") or real.get("ci_excludes_chance"))
    if not control_ok:
        verdict = "harness_unreliable"
    elif _ok(real_auc) and real_auc > 0.6 and real_ci_ok:
        verdict = "labels_learnable_offline"
    else:
        verdict = "labels_not_learnable_from_features"
    report["verdict"] = verdict
    report["verdict_detail"] = {
        "positive_control_auc": control_auc,
        "positive_control_raw_auc": raw_control.get("auc"),
        "positive_control_ok": control_ok,
        "token_content_auc": real_auc,
        "token_content_auc_ci_excludes_chance": real_ci_ok,
        "context_only_auc": context_only.get("auc"),
        "interpretation": {
            "harness_unreliable": (
                "the positive control failed, so this run cannot support any "
                "conclusion about the real labels"
            ),
            "labels_learnable_offline": (
                "offline supervision recovers the label from features the "
                "selector sees; the online loss is the thing to fix"
            ),
            "labels_not_learnable_from_features": (
                "offline supervision cannot recover the label either, so no "
                "amount of selector capacity or training will help; the label "
                "construction must change"
            ),
        }.get(verdict),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    for key, value in report["results"].items():
        if "auc" not in value:
            print(f"{key:44s} {json.dumps(value)}")
            continue
        value_auc = value["auc"]
        print(
            f"{key:44s} n={value['n']:6d} AUC={value_auc:.4f} "
            f"CI[{value['auc_ci95'][0]:.3f},{value['auc_ci95'][1]:.3f}] p={value['permutation_p_value']:.4f}"
        )
    for key, value in report.get("within_scene_ranking", {}).items():
        if not isinstance(value, dict):
            continue
        print(
            f"ranking::{key:36s} pairwise={value['pairwise_accuracy']:.4f} "
            f"pairs={value['n_pairs']:6d} scenes={value['n_scenes']:5d} "
            f"mean_rho={value['mean_within_scene_spearman']:.4f}"
        )
    for key, value in report.get("dead_zone_sweep", {}).items():
        if value.get("skipped"):
            print(f"{key:44s} skipped n_train={value['n_train']} n_eval={value['n_eval']}")
            continue
        print(
            f"{key:44s} n={value['n']:6d} AUC={value['auc']:.4f} "
            f"CI[{value['auc_ci95'][0]:.3f},{value['auc_ci95'][1]:.3f}] p={value['permutation_p_value']:.4f} "
            f"abstain={value['abstain_ratio_train']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
