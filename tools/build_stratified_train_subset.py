#!/usr/bin/env python3
import csv, json, math, pickle, shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 42
TARGET = 2048
SRC = Path('/mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework/outputs/navsim_split_audit/metadata/train')
OUT = Path('/mnt/nvme/xiangyike/DriveVA-official-base/outputs/train_stratified_2048_seed42')
Y_P90 = 2.51031082
HEADING_P90 = 0.28744094
SPEED_Q1 = 0.03208101
SPEED_Q2 = 5.98822530


def features(path):
    frames = [x for x in pickle.load(open(path, 'rb')) if isinstance(x, dict)]
    if len(frames) < 9:
        raise ValueError(f'{path} has only {len(frames)} frames')
    t0 = np.asarray(frames[0]['ego2global'], dtype=float)
    R0, p0 = t0[:3, :3], t0[:3, 3]
    positions = np.asarray([np.asarray(x['ego2global'], dtype=float)[:3, 3] for x in frames[:9]])
    local = (positions - p0) @ R0
    yaw0 = math.atan2(R0[1, 0], R0[0, 0])
    R8 = np.asarray(frames[8]['ego2global'], dtype=float)[:3, :3]
    yaw8 = math.atan2(R8[1, 0], R8[0, 0])
    heading = math.atan2(math.sin(yaw8 - yaw0), math.cos(yaw8 - yaw0))
    speed = float(np.linalg.norm(np.asarray(frames[0]['ego_dynamic_state'], dtype=float)[:2]))
    endpoint = local[8, :2]
    # Signed curvature proxy: heading change divided by forward endpoint distance.
    curvature = float(heading / max(abs(float(endpoint[0])), 1e-3))
    if endpoint[1] >= Y_P90 or heading >= HEADING_P90:
        turn = 'left'
    elif endpoint[1] <= -Y_P90 or heading <= -HEADING_P90:
        turn = 'right'
    else:
        turn = 'straight'
    if speed <= SPEED_Q1:
        speed_bin = 'low'
    elif speed <= SPEED_Q2:
        speed_bin = 'medium'
    else:
        speed_bin = 'high'
    first = frames[0]
    log_name = str(first.get('log_name', path.stem))
    sample_id = str(first.get('token', path.stem))
    return dict(sample_id=sample_id, log_name=log_name, source_file=path.name,
                speed=speed, turn_class=turn, speed_bin=speed_bin,
                strata=f'{turn}__{speed_bin}', endpoint_x=float(endpoint[0]),
                endpoint_y=float(endpoint[1]), heading_change=float(heading),
                curvature=curvature)


def proportional_counts(counts, total):
    keys = sorted(counts)
    raw = {k: counts[k] * total / sum(counts.values()) for k in keys}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    for k in sorted(keys, key=lambda k: (raw[k] - out[k], k), reverse=True)[:total - sum(out.values())]:
        out[k] += 1
    return out


def stats(rows):
    def ms(key):
        a = np.asarray([r[key] for r in rows], dtype=float)
        return {'mean': float(a.mean()), 'std': float(a.std()),
                'min': float(a.min()), 'p25': float(np.percentile(a, 25)),
                'p50': float(np.percentile(a, 50)), 'p75': float(np.percentile(a, 75)),
                'p90': float(np.percentile(a, 90)), 'p95': float(np.percentile(a, 95)),
                'max': float(a.max())}
    return {'count': len(rows), 'speed': ms('speed'),
            'endpoint_x': ms('endpoint_x'), 'endpoint_y': ms('endpoint_y'),
            'heading_change': ms('heading_change'), 'curvature': ms('curvature'),
            'strata': dict(sorted(Counter(r['strata'] for r in rows).items()))}


def main():
    paths = sorted(SRC.glob('*.pkl'))
    rows = [features(p) for p in paths]
    if len(rows) != 3768:
        raise RuntimeError(f'Expected 3768 source logs, got {len(rows)}')
    source_counts = Counter(r['strata'] for r in rows)
    target_counts = proportional_counts(source_counts, TARGET)
    rng = np.random.default_rng(SEED)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r['strata']].append(r)
    selected = []
    for key in sorted(buckets):
        idx = rng.choice(len(buckets[key]), size=target_counts[key], replace=False)
        selected.extend(buckets[key][int(i)] for i in idx)
    selected.sort(key=lambda r: r['source_file'])
    if len(selected) != TARGET or len({r['source_file'] for r in selected}) != TARGET:
        raise RuntimeError('Selection is not exactly 2048 unique files')
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob('*.pkl'):
        old.unlink()
    for r in selected:
        shutil.copy2(SRC / r['source_file'], OUT / r['source_file'])
    for r in selected:
        r['random_seed'] = SEED
    audit = {
        'source_dir': str(SRC), 'output_dir': str(OUT), 'seed': SEED,
        'source_count': len(rows), 'selected_count': len(selected),
        'thresholds': {'speed_q1': SPEED_Q1, 'speed_q2': SPEED_Q2,
                       'turn_endpoint_abs_p90_m': Y_P90,
                       'turn_abs_heading_change_p90_rad': HEADING_P90,
                       'turn_rule': 'left if endpoint_y>=threshold OR heading>=threshold; right if both negative; else straight'},
        'source': stats(rows), 'selected': stats(selected),
        'source_log_unique': len({r['log_name'] for r in rows}) == len(rows),
        'selected_log_unique': len({r['log_name'] for r in selected}) == len(selected),
        'source_unique_log_count': len({r['log_name'] for r in rows}),
        'selected_unique_log_count': len({r['log_name'] for r in selected}),
        'source_duplicate_log_entries': len(rows) - len({r['log_name'] for r in rows}),
        'selected_duplicate_log_entries': len(selected) - len({r['log_name'] for r in selected}),
        'selected_sample_unique': len({r['sample_id'] for r in selected}) == len(selected),
        'strata_target_counts': dict(sorted(target_counts.items())),
    }
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    with (OUT / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['sample_id', 'log_name', 'source_file', 'speed', 'turn_class', 'speed_bin', 'strata', 'endpoint_x', 'endpoint_y', 'heading_change', 'curvature', 'random_seed']
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(selected)
    print(json.dumps({'output_dir': str(OUT), 'source': audit['source']['strata'], 'selected': audit['selected']['strata'], 'thresholds': audit['thresholds']}, indent=2))


if __name__ == '__main__':
    main()
