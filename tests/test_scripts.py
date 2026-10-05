import os
import subprocess
import time

import pytest

LIB = "scripts/qwen/lib.sh"
SCRIPTS = ["scripts/qwen/lib.sh", "scripts/qwen/ced/run_ced_v2.sh", "scripts/qwen/ced/run_cllora.sh",
           "scripts/qwen/ced/run_all_cllora.sh", "scripts/qwen/cre/run_cre_cllora.sh",
           "scripts/qwen/cre/run_cre_dist.sh"]


def bash(cmd):
    return subprocess.run(["bash", "-c", f"source {LIB}; {cmd}"], capture_output=True, text=True)


@pytest.mark.parametrize("args,expected", [
    ("2 16 16", "16 2"),      # the H200 case
    ("2 16 12", "8 4"),       # 12 is not a divisor of the accumulation: largest valid below it
    ("32 1 16", "32 1"),      # target below the logical micro-batch: stay logical
    ("2 16", "2 16"),         # no target: unchanged
    ("16 2 64", "32 1"),      # capped by the accumulation
    ("2 16 1000", "32 1"),
])
def test_phys_split_keeps_the_logical_batch_and_rows_per_update(args, expected):
    assert bash(f"phys_split {args}").stdout.strip() == expected


def test_run_active_only_matches_the_same_method_and_order():
    fake = subprocess.Popen(["bash", "-c", "exec -a 'python cl_lora/engine.py --cl-method tree "
                                           "--data-root data/fewrel_perm1 --num-tasks 10' sleep 30"])
    try:
        time.sleep(0.5)
        assert bash("run_active tree data/fewrel_perm1").returncode == 0
        assert bash("run_active tree data/fewrel_perm0").returncode == 1
        assert bash("run_active inclora data/fewrel_perm1").returncode == 1
    finally:
        fake.kill()


def test_resuming_a_run_started_with_another_config_is_refused(tmp_path):
    (tmp_path / "data").mkdir()
    out = tmp_path / "run" / "run_manifest.json"
    base = ["python3", "tools/ced_run_manifest.py", "init", "--output", str(out), "--run", "r",
            "--method", "rkl", "--permutation", "0", "--seed", "42", "--data-root", str(tmp_path / "data"),
            "--runtime-file", "ced_finetune.py", "--model", "m", "--rank", "16", "--alpha", "64",
            "--dropout", "0.1", "--gpu-count", "1", "--micro-batch", "2", "--gradient-accumulation", "16",
            "--epochs", "5"]
    assert subprocess.run(base + ["--config", "mode=ce_kd"]).returncode == 0
    resumed = subprocess.run(base + ["--config", "mode=ce_kd;phys=16x2", "--resume"], capture_output=True, text=True)
    assert resumed.returncode != 0
    assert "resume manifest mismatch for config" in resumed.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_parse(script):
    assert subprocess.run(["bash", "-n", script]).returncode == 0
