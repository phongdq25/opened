import os
import subprocess


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
