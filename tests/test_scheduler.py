import os
import signal
import subprocess

# the launch settings a test's environment must not inherit from the shell that runs pytest
SETTINGS = ("POOL_GPUS", "SLOTS_PER_GPU", "PHYS_BS", "USE_MPS", "NEED_GPU_MB", "NEED_LORA_MB", "GEN_BACKEND",
            "COMPILE_GEN", "GRAD_CKPT", "OURS_PHYS_BS", "OURS_GRAD_CKPT", "ONLY")


def test_dry_run_schedules_cre_methods_after_their_shared_task0():
    log = "logs/fewrel_matrix_pool.log"
    if os.path.exists(log):
        os.remove(log)
    env = {**os.environ, "DRY": "1", "DS": "fewrel", "PERMS": "0", "POOL_GPUS": "0", "SLOTS_PER_GPU": "3",
           "SKIP_INSTALL": "1"}
    out = subprocess.run(["bash", "project_commands.sh"], env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    lines = open(log).read().splitlines()
    assert not any("FAILED" in line for line in lines)
    done = [line for line in lines if " done " in line]
    assert len(done) == 16                                   # 1 shared task0 + 7 distillation + 8 CL-LoRA
    t0_done = next(i for i, line in enumerate(lines) if " done " in line and "b_dist_t0/" in line)
    method_starts = [i for i, line in enumerate(lines)
                     if " start " in line and "b_dist_" in line and "b_dist_t0/" not in line]
    assert method_starts and min(method_starts) > t0_done
    cl_starts = [i for i, line in enumerate(lines) if " start " in line and "b_cllora_" in line]
    assert len(cl_starts) == 8


def train_step_line(tmp_path, card_mib):
    """The scheduler's '3. train' line on a fake host with two cards of card_mib MiB."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text(f'#!/bin/bash\ncase "$*" in *memory.total*) printf "{card_mib}\\n{card_mib}\\n" ;; '
                   '*index*) printf "0\\n1\\n" ;; esac\n')
    smi.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in SETTINGS}
    vllm_py = bin_dir / "vllm_python"            # a vLLM environment for gen_backend.find_vllm_python()
    vllm_py.write_text('#!/bin/bash\nif [ "$1" = "-c" ]; then echo "VLLM_VERSION 0.27.1"; exit 0; fi\n')
    vllm_py.chmod(0o755)
    env.update(DRY="1", DS="fewrel", PERMS="0", SKIP_INSTALL="1", PATH=f"{bin_dir}:{env['PATH']}", VLLM_PY=str(vllm_py))
    proc = subprocess.Popen(["bash", "project_commands.sh"], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        for line in proc.stdout:
            if line.startswith("=== 3. train"):
                return line.strip()
        return None
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def test_h200_class_cards_get_the_measured_packing_spread_over_the_cards(tmp_path):
    assert train_step_line(tmp_path, 143771) == \
        "=== 3. train 16 jobs on gpus 0 1 0 1 0 1 (PHYS_BS=8 USE_MPS=1 GEN_BACKEND=vllm COMPILE_GEN=0 GRAD_CKPT=0) ==="


def test_smaller_cards_keep_one_run_per_gpu_and_the_runner_defaults(tmp_path):
    assert train_step_line(tmp_path, 46068) == \
        "=== 3. train 16 jobs on gpus 0 1 (PHYS_BS=runner default USE_MPS=0 GEN_BACKEND=hf COMPILE_GEN=0 GRAD_CKPT=0) ==="


def dry_plan(**settings):
    """A DRY=1 launch on this host's data: its '3. train' line and its plan, {job name: plan line},
    read without waiting for the stub scheduler."""
    env = {k: v for k, v in os.environ.items() if k not in SETTINGS}
    env.update(DRY="1", PERMS="0", POOL_GPUS="0", SLOTS_PER_GPU="3", SKIP_INSTALL="1", **settings)
    proc = subprocess.Popen(["bash", "project_commands.sh"], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    step_line, plan = None, {}
    try:
        for line in proc.stdout:
            if line.startswith("=== 3. train"):
                step_line = line.strip()
            elif line.startswith("  ") and " perm0 " in line:
                plan[line.split()[0]] = line.strip()
            elif line.startswith("DRY=1: no training"):
                break
        return step_line, plan
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def test_ours_runs_take_their_own_batch_and_checkpointing():
    # OURS_PHYS_BS / OURS_GRAD_CKPT reach Ours and its ablations only: the baselines of the same
    # launch, and the plain-SFT task0 Ours starts from, keep PHYS_BS / GRAD_CKPT
    step_line, plan = dry_plan(DS="fewrel", ONLY="g1 b", PHYS_BS="16", OURS_PHYS_BS="32", OURS_GRAD_CKPT="1")
    assert step_line.endswith("GRAD_CKPT=0; Ours PHYS_BS=32 GRAD_CKPT=1) ===")
    assert plan["g1_full"].endswith("[PHYS_BS=32 GRAD_CKPT=1]")
    for job in ("task0", "b_dist_t0", "b_dist_rkl", "b_cllora_tree"):
        assert plan[job].endswith("[PHYS_BS=16 GRAD_CKPT=0]"), plan[job]


def test_without_ours_settings_ours_runs_take_the_launch_settings():
    step_line, plan = dry_plan(DS="fewrel", ONLY="g1", PHYS_BS="16", GRAD_CKPT="1")
    assert step_line.endswith("COMPILE_GEN=0 GRAD_CKPT=1) ===")
    assert plan["g1_full"].endswith("[PHYS_BS=16 GRAD_CKPT=1]")


def test_the_scheduler_starts_ours_runs_with_their_settings():
    # DRY=1 never reaches launch(), so pin the line that hands the settings to ours_queue.sh
    assert 'env $(job_settings "${name}" "${sd}") bash scripts/qwen/ced/ours_queue.sh' in open("project_commands.sh").read()
