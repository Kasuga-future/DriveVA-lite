"""Protocol labelling must be automatic and impossible to get silently wrong.

Absolute PDM is only comparable to published NAVSIM numbers on the primary
protocol (``navtest-7876``).  The project silently switched to a harder 1,920
scene subset for a while, which moved every absolute number by about one point
while leaving paired deltas intact (see
``reports/eval_protocol_baseline_discrepancy_20260911.md``).  These tests pin the
resolution rules that keep every run self-describing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "run_official_navsim_press", FRAMEWORK_ROOT / "scripts" / "run_official_navsim_press.py"
)
press = importlib.util.module_from_spec(_spec)
sys.modules["run_official_navsim_press"] = press
_spec.loader.exec_module(press)


def _args(**overrides) -> argparse.Namespace:
    values = {key: None for key in press._PATH_KEYS}
    values.update({"eval_protocol": "auto"})
    values.update(overrides)
    return argparse.Namespace(**values)


def test_primary_protocol_is_the_paper_comparable_navtest_7876() -> None:
    assert press.PRIMARY_EVAL_PROTOCOL == "navtest-7876"
    preset = press.EVAL_PROTOCOLS["navtest-7876"]
    assert preset["expected_scenes"] == 7876
    # the published reference point this project must quote
    assert abs(preset["baselines"]["no_press_pdm"] - 0.909839) < 5e-7


def test_auto_with_no_paths_uses_the_primary_protocol() -> None:
    resolved = press.resolve_eval_protocol(_args())
    assert resolved["label"] == "navtest-7876"
    assert resolved["overridden"] is False
    assert resolved["navsim_log_path"] == press.EVAL_PROTOCOLS["navtest-7876"]["navsim_log_path"]


def test_auto_recognises_the_harder_split_subset() -> None:
    preset = press.EVAL_PROTOCOLS["split-test-1920"]
    resolved = press.resolve_eval_protocol(
        _args(**{key: preset[key] for key in press._PATH_KEYS})
    )
    assert resolved["label"] == "split-test-1920"
    assert resolved["overridden"] is False


def test_partial_paths_are_completed_from_the_primary_protocol() -> None:
    resolved = press.resolve_eval_protocol(
        _args(metric_cache_path=Path("/tmp/somewhere/metric_cache"))
    )
    assert resolved["label"] == "navtest-7876+overridden"
    assert resolved["overridden"] is True
    # the unspecified paths still come from the preset
    assert resolved["scene_filter_yaml"] == press.EVAL_PROTOCOLS["navtest-7876"]["scene_filter_yaml"]


def test_unknown_data_paths_are_labelled_custom() -> None:
    resolved = press.resolve_eval_protocol(
        _args(**{key: Path(f"/tmp/unknown/{key}") for key in press._PATH_KEYS})
    )
    assert resolved["label"] == "custom"
    assert resolved["preset"] is None


def test_explicit_preset_choice_is_honoured() -> None:
    resolved = press.resolve_eval_protocol(_args(eval_protocol="split-test-1920"))
    assert resolved["label"] == "split-test-1920"
    assert resolved["metric_cache_path"] == press.EVAL_PROTOCOLS["split-test-1920"]["metric_cache_path"]


def test_unknown_preset_name_is_rejected() -> None:
    try:
        press.resolve_eval_protocol(_args(eval_protocol="does-not-exist"))
    except ValueError as exc:
        assert "unknown eval protocol" in str(exc)
    else:
        raise AssertionError("an unknown protocol name must be rejected, not silently defaulted")


def test_apply_eval_protocol_writes_paths_back_and_labels_baselines() -> None:
    args = _args()
    resolved = press.apply_eval_protocol(args)
    assert resolved["label"] == "navtest-7876"
    assert args.eval_protocol_label == "navtest-7876"
    assert args.eval_protocol_expected_scenes == 7876
    assert abs(args.eval_protocol_baselines["no_press_pdm"] - 0.909839) < 5e-7
    for key in press._PATH_KEYS:
        assert getattr(args, key) is not None


# ---------------------------------------------------------------------------
# BUG-3: a truncated run must be self-describing AND must not exit 0.
# ---------------------------------------------------------------------------


def _scoped_args(**overrides) -> argparse.Namespace:
    args = _args(**overrides)
    press.apply_eval_protocol(args)
    return args


def _summary(valid_scenes: int, method: str = "m") -> dict:
    return {"method_rows": [{"method": method, "valid_scenes": valid_scenes}]}


def test_evaluation_scope_flags_truncated_and_full_runs() -> None:
    full = press.evaluation_scope(_scoped_args())
    assert full["is_truncated"] is False
    assert full["max_eval_tokens"] is None
    assert full["expected_scenes"] == 7876
    assert full["enable_nuscenes_metrics"] is False

    truncated = press.evaluation_scope(_scoped_args(max_eval_tokens=8))
    assert truncated["is_truncated"] is True
    assert truncated["max_eval_tokens"] == 8
    assert truncated["expected_scenes"] == 7876

    # A cap at or above the protocol size removes nothing, so it is not a
    # truncation even though the flag was supplied.
    assert press.evaluation_scope(_scoped_args(max_eval_tokens=7876))["is_truncated"] is False
    assert press.evaluation_scope(_scoped_args(max_eval_tokens=8000))["is_truncated"] is False


def test_custom_protocol_with_a_cap_counts_as_truncated() -> None:
    args = argparse.Namespace(
        **{key: Path(f"/tmp/unknown/{key}") for key in press._PATH_KEYS}
    )
    args.eval_protocol = "custom"
    args.max_eval_tokens = 8
    press.apply_eval_protocol(args)
    # No declared size -> truncation cannot be excluded.
    assert args.eval_protocol_expected_scenes is None
    assert press.is_evaluation_truncated(args) is True
    assert press.evaluation_scope(args)["is_truncated"] is True


def test_method_config_dict_records_evaluation_scope(tmp_path: Path) -> None:
    args = _scoped_args(max_eval_tokens=8)
    args.repo_root = tmp_path
    args.full_ckpt = tmp_path / "full.safetensors"
    args.local_model_path = tmp_path / "models"
    args.num_inference_steps = 3
    args.model_future_frames = 8
    args.seed = 0
    args.sample_seed = None
    args.dump_trajectories = False
    args.retention_policy = None
    args.poc_test_derived = True
    args.score_cache_root = None
    config = press.method_config_dict(
        args, {"name": "m", "mode": "physical", "press": {"domain": "history"}}, 1
    )
    scope = config["evaluation_scope"]
    assert scope["is_truncated"] is True
    assert scope["max_eval_tokens"] == 8
    assert scope["expected_scenes"] == 7876
    # The scope block must survive the JSON round trip the artifacts use.
    assert json.loads(json.dumps(config))["evaluation_scope"] == scope

    full_args = _scoped_args()
    full_args.repo_root = tmp_path
    full_args.full_ckpt = tmp_path / "full.safetensors"
    full_args.local_model_path = tmp_path / "models"
    full_args.num_inference_steps = 3
    full_args.model_future_frames = 8
    full_args.seed = 0
    full_args.sample_seed = None
    full_args.dump_trajectories = False
    full_args.retention_policy = None
    full_args.poc_test_derived = False
    full_args.score_cache_root = None
    full_config = press.method_config_dict(
        full_args, {"name": "m", "mode": "physical", "press": {"domain": "history"}}, 1
    )
    assert full_config["evaluation_scope"]["is_truncated"] is False


def test_truncated_run_is_not_citable_and_full_run_is() -> None:
    truncated = _scoped_args(max_eval_tokens=8)
    # Truncated runs fail even in the impossible case where they produced the
    # full scene count: the flag alone disqualifies the absolute number.
    assert press.enforce_evaluation_scope(truncated, _summary(8)) == 1
    assert press.enforce_evaluation_scope(truncated, _summary(7876)) == 1

    full = _scoped_args()
    assert press.enforce_evaluation_scope(full, _summary(7876)) == 0
    # A full run with a short valid count (silently invalid scenes) also fails.
    assert press.enforce_evaluation_scope(full, _summary(7875)) == 1


def test_scope_gate_sets_the_real_process_exit_code() -> None:
    """`run()` returns `enforce_evaluation_scope(...)`, so pin the exit status.

    This runs the scope gate in a real subprocess (the same way the CLI does)
    instead of only checking the in-process return value.  It does not run an
    evaluation; it pins the exit-code path that makes a truncated run fail.
    """
    code = (
        "import argparse, importlib.util, sys\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "spec = importlib.util.spec_from_file_location(\n"
        "    'run_official_navsim_press', root / 'scripts' / 'run_official_navsim_press.py')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['run_official_navsim_press'] = module\n"
        "spec.loader.exec_module(module)\n"
        "def args(max_tokens):\n"
        "    return argparse.Namespace(eval_protocol_label='navtest-7876',\n"
        "        eval_protocol_expected_scenes=7876, max_eval_tokens=max_tokens)\n"
        "def summary(valid):\n"
        "    return dict(method_rows=[dict(method='m', valid_scenes=valid)])\n"
        "codes = (\n"
        "    module.enforce_evaluation_scope(args(8), summary(8)),\n"
        "    module.enforce_evaluation_scope(args(8), summary(7876)),\n"
        "    module.enforce_evaluation_scope(args(None), summary(7876)),\n"
        "    module.enforce_evaluation_scope(args(None), summary(7875)),\n"
        ")\n"
        "print('codes', codes)\n"
        "sys.exit(0 if codes == (1, 1, 0, 1) else 3)\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": f"{FRAMEWORK_ROOT}:{FRAMEWORK_ROOT.parent}",
        "CUDA_VISIBLE_DEVICES": "",
    }
    proc = subprocess.run(
        [sys.executable, "-c", code, str(FRAMEWORK_ROOT)],
        cwd=str(FRAMEWORK_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "codes (1, 1, 0, 1)" in proc.stdout


def test_official_loader_disables_unused_lidar_for_camera_only_driveva() -> None:
    """Trajectory-only evaluation must not require absent point-cloud files."""

    official = press._load_official_eval_module()

    class FakeSensorConfig:
        @classmethod
        def build_all_sensors(cls, include=True):
            assert include is True
            return argparse.Namespace(cam_f0=True, lidar_pc=True)

    captured = {}

    class FakeSceneLoader:
        def __init__(
            self,
            data_path,
            sensor_blobs_path,
            scene_filter,
            sensor_config,
            load_image_path,
        ):
            captured["sensor_config"] = sensor_config

    args = argparse.Namespace(
        navsim_log_path=Path("/tmp/logs"),
        sensor_blobs_path=Path("/tmp/sensors"),
    )
    official._build_scene_loader(FakeSceneLoader, FakeSensorConfig, args, object())
    assert captured["sensor_config"].cam_f0 is True
    assert captured["sensor_config"].lidar_pc is False


def _load_verify_module():
    spec = importlib.util.spec_from_file_location(
        "verify_eval_protocol", FRAMEWORK_ROOT / "scripts" / "verify_eval_protocol.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_eval_protocol"] = module
    spec.loader.exec_module(module)
    return module


def _fake_run_root(root: Path, *, is_truncated: bool, n_rows: int) -> Path:
    preset = press.EVAL_PROTOCOLS["navtest-7876"]
    method = root / "round01" / "physical_test_method"
    method.mkdir(parents=True)
    (root / "suite_manifest.json").write_text(
        json.dumps(
            {
                "data": {key: str(preset[key]) for key in press._PATH_KEYS},
                "eval_protocol": {"label": "navtest-7876", "expected_scenes": 7876},
                "evaluation_scope": {
                    "max_eval_tokens": 8 if is_truncated else None,
                    "force_full_scene_set": False,
                    "enable_nuscenes_metrics": True,
                    "expected_scenes": 7876,
                    "is_truncated": is_truncated,
                },
            }
        ),
        encoding="utf-8",
    )
    rows = ["token,valid,pdm_score"]
    rows += [f"scene-{i},True,{0.9 + (i % 7) * 1e-6:.9f}" for i in range(n_rows)]
    rows.append("average,True,0.9")
    (method / "pdm_score_0.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def test_verify_eval_protocol_rejects_a_truncated_run(tmp_path: Path, monkeypatch) -> None:
    verify = _load_verify_module()
    root = _fake_run_root(tmp_path / "truncated", is_truncated=True, n_rows=8)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_eval_protocol.py", str(root), "--expect-protocol", "navtest-7876"],
    )
    assert verify.main() == 1


def test_verify_eval_protocol_accepts_an_untruncated_full_run(
    tmp_path: Path, monkeypatch
) -> None:
    verify = _load_verify_module()
    root = _fake_run_root(tmp_path / "full", is_truncated=False, n_rows=7876)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_eval_protocol.py", str(root), "--expect-protocol", "navtest-7876"],
    )
    assert verify.main() == 0
