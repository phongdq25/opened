import os
import subprocess
import sys
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


def fake_bin(tmp_path, **scripts):
    """Executables named after the keyword arguments, put first on PATH by the callers."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in scripts.items():
        path = bin_dir / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    return bin_dir


def fake_smi(mib):
    """nvidia-smi on a host with two cards of `mib` MiB each."""
    return f'case "$*" in *memory.total*) printf "{mib}\\n{mib}\\n" ;; *index*) printf "0\\n1\\n" ;; esac'


@pytest.mark.parametrize("mib,expected", [
    ("143771", "3 8 1 39833 18432 vllm 0"),  # H200 NVL: the values tools/bench_gpu.sh measured
    ("46068", "1 - 0 - - hf 0"),           # 46 GB cards: one run per GPU, the runners' own settings
    ("0", "1 - 0 - - hf 0"),               # no nvidia-smi
])
def test_gpu_defaults_give_the_h200_measurements_only_to_h200_class_cards(mib, expected):
    assert bash(f"gpu_defaults {mib}").stdout.strip() == expected


def vllm_python(tmp_path, version="0.27.1", gpu=True):
    """A VLLM_PY that tells gen_backend.find_vllm_python() it holds this vLLM version, and
    gen_backend.vllm_sees_gpu() whether its torch can use the GPU."""
    python = tmp_path / "vllm_python"
    python.write_text(f'#!/bin/bash\ncase "$2" in *cuda.is_available*) exit {0 if gpu else 1} ;; esac\n'
                      f'if [ "$1" = "-c" ]; then echo "VLLM_VERSION {version}"; exit 0; fi\n')
    python.chmod(0o755)
    return str(python)


@pytest.mark.parametrize("mib,preset,vllm,expected", [
    ("143771", "", "found", "slots=3 phys=8 mps=1 gpu=39833 lora=18432 gen=vllm cgen=0"),
    ("143771", "", "missing", "slots=3 phys=8 mps=1 gpu=39833 lora=18432 gen=hf cgen=0"),
    ("143771", "", "untested", "slots=3 phys=8 mps=1 gpu=39833 lora=18432 gen=hf cgen=0"),
    ("143771", "", "no_gpu", "slots=3 phys=8 mps=1 gpu=39833 lora=18432 gen=hf cgen=0"),  # driver too old
    ("46068", "", "found", "slots=1 phys=unset mps=0 gpu=unset lora=unset gen=hf cgen=0"),
    ("143771", "PHYS_BS=16 SLOTS_PER_GPU=2 USE_MPS=0 GEN_BACKEND=hf COMPILE_GEN=1", "found",
     "slots=2 phys=16 mps=0 gpu=39833 lora=18432 gen=hf cgen=1"),
])
def test_apply_card_defaults_fills_only_what_the_caller_left_unset(tmp_path, mib, preset, vllm, expected):
    bin_dir = fake_bin(tmp_path, **{"nvidia-smi": fake_smi(mib)})
    preset = f"export {preset};" if preset else ""          # what the caller's environment already holds
    vllm_py = {"found": lambda: vllm_python(tmp_path), "untested": lambda: vllm_python(tmp_path, "0.30.0"),
               "no_gpu": lambda: vllm_python(tmp_path, gpu=False), "missing": lambda: str(tmp_path / "missing")}[vllm]()
    out = subprocess.run(["bash", "-c", f'source {os.path.abspath(LIB)}; {preset} apply_card_defaults 0 1; '
                          'echo "slots=$SLOTS_PER_GPU phys=${PHYS_BS:-unset} mps=$USE_MPS '
                          'gpu=${NEED_GPU_MB:-unset} lora=${NEED_LORA_MB:-unset} gen=${GEN_BACKEND:-unset} '
                          'cgen=${COMPILE_GEN:-unset}"'],
                         env={k: v for k, v in {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                                                "VLLM_PY": vllm_py, "PY": sys.executable}.items()
                              if k not in ("PHYS_BS", "SLOTS_PER_GPU", "USE_MPS", "NEED_GPU_MB", "NEED_LORA_MB",
                                           "GEN_BACKEND", "COMPILE_GEN")},
                         capture_output=True, text=True)
    assert out.stdout.strip() == expected, out.stderr


# A daemon per pipe directory, as the real one: -d takes 0.5 s and refuses a second one, the daemon
# outlives the call with whatever descriptors it inherited, get_server_list fails without one, quit ends it.
MPS_CONTROL = '''up="${MPS_LOG}.up"
if [ $# -gt 0 ]; then
    echo "args:$*" >> "${MPS_LOG}"; sleep 0.5; [ -e "${up}" ] && exit 1; touch "${up}"
    sleep 600 > /dev/null 2>&1 < /dev/null &
    echo $! > "${MPS_LOG}.pid"; exit 0
fi
cmd=$(cat); echo "stdin:${cmd}" >> "${MPS_LOG}"
case ${cmd} in
    get_server_list) [ -e "${up}" ] ;;
    quit) rm -f "${up}"; kill "$(cat "${MPS_LOG}.pid")" 2> /dev/null; true ;;
esac'''


def stop_fake_daemon(log):
    pid = f"{log}.pid"
    if os.path.exists(pid):
        subprocess.run(["kill", open(pid).read().strip()], capture_output=True)


def line(proc, timeout=20):
    """The next line proc prints, or None when it prints nothing for timeout seconds."""
    import select
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    return proc.stdout.readline().strip() if ready else None


@pytest.mark.parametrize("client_rc", [0, 1])
def test_mps_start_keeps_the_daemon_only_when_a_cuda_client_can_use_it(tmp_path, client_rc):
    bin_dir = fake_bin(tmp_path, **{"nvidia-cuda-mps-control": MPS_CONTROL, "fakepy": f"exit {client_rc}"})
    log = tmp_path / "mps.log"
    out = subprocess.run(["bash", "-c", f'source {os.path.abspath(LIB)}; cd {tmp_path}; mps_start {bin_dir}/fakepy 0; '
                          'echo "rc=$? pipe=${CUDA_MPS_PIPE_DIRECTORY:-unset}"'],
                         env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "MPS_LOG": str(log)},
                         capture_output=True, text=True)
    stop_fake_daemon(log)
    calls = log.read_text()
    assert "args:-d" in calls
    if client_rc == 0:
        assert out.stdout.strip() == f"rc=0 pipe={tmp_path}/.mps/pipe" and "quit" not in calls
    else:
        assert out.stdout.strip() == "rc=1 pipe=unset" and "stdin:quit" in calls



def launch_shell(tmp_path, bin_dir, log, name):
    """A bash 'launch' that joins CUDA MPS, says so, and leaves it once tmp_path/<name>.stop appears."""
    stop = tmp_path / f"{name}.stop"
    return subprocess.Popen(
        ["bash", "-c", f'source {os.path.abspath(LIB)}; cd {tmp_path}; mps_start {bin_dir}/fakepy 0 && echo joined || echo failed; '
                       f'while [ ! -e {stop} ]; do sleep 0.1; done; mps_stop; echo left'],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "MPS_LOG": str(log)},
        stdout=subprocess.PIPE, text=True)


def test_launches_share_one_mps_daemon_and_the_last_one_out_stops_it(tmp_path):
    bin_dir = fake_bin(tmp_path, **{"nvidia-cuda-mps-control": MPS_CONTROL, "fakepy": "exit 0"})
    log = tmp_path / "mps.log"
    first = launch_shell(tmp_path, bin_dir, log, "first")
    second = None
    try:
        assert line(first) == "joined"
        second = launch_shell(tmp_path, bin_dir, log, "second")
        assert line(second) == "joined"
        assert log.read_text().count("args:-d") == 1              # the second joined the first one's daemon
        (tmp_path / "first.stop").touch()
        assert line(first) == "left" and first.wait(timeout=30) == 0
        assert "stdin:quit" not in log.read_text()                 # the second still runs jobs on it
        (tmp_path / "second.stop").touch()
        assert line(second) == "left" and second.wait(timeout=30) == 0
        assert log.read_text().count("stdin:quit") == 1            # the daemon held no launch's lock
    finally:
        for launch in (first, second):
            if launch is not None:
                launch.kill()
        stop_fake_daemon(log)


def test_launches_starting_together_start_one_mps_daemon_and_all_join_it(tmp_path):
    # the fake daemon takes 0.5 s to come up: a launch must not miss it and start a second one
    bin_dir = fake_bin(tmp_path, **{"nvidia-cuda-mps-control": MPS_CONTROL, "fakepy": "exit 0"})
    log = tmp_path / "mps.log"
    launches = [launch_shell(tmp_path, bin_dir, log, name) for name in ("a", "b", "c")]
    try:
        # the daemon outlives the launch that started it: it must not keep that launch's start lock
        assert [line(launch) for launch in launches] == ["joined"] * 3
        assert log.read_text().count("args:-d") == 1
        for name, launch in zip(("a", "b", "c"), launches):
            (tmp_path / f"{name}.stop").touch()
            assert line(launch) == "left" and launch.wait(timeout=30) == 0
        assert log.read_text().count("stdin:quit") == 1
    finally:
        for launch in launches:
            launch.kill()
        stop_fake_daemon(log)


def test_a_slot_held_by_a_running_job_is_not_taken_until_the_job_exits(tmp_path):
    slots = tmp_path / "slots"
    lock = lambda gpu, k: bash(f"slot_lock {slots} {gpu} {k}").returncode
    # what the scheduler does: lock the slot, start the job (it inherits the lock), close its own copy
    holder = subprocess.Popen(["bash", "-c", f"source {LIB}; slot_lock {slots} 0 1 || exit 9; "
                               "sleep 60 & job=$!; slot_unlock; echo $job; wait $job"],
                              stdout=subprocess.PIPE, text=True)
    job = int(holder.stdout.readline())
    try:
        assert lock(0, 1) == 1                                     # held by the job, not by its launch
        assert lock(0, 2) == 0 and lock(1, 1) == 0                 # other slots stay free
    finally:
        os.kill(job, 15)
        holder.wait(timeout=30)
    assert lock(0, 1) == 0

def test_run_sh_takes_its_defaults_from_the_cards():
    assert "apply_card_defaults" in open("run.sh").read()


@pytest.mark.parametrize("backend,found,rc", [("hf", False, 0), ("vllm", False, 1), ("vllm", True, 0)])
def test_vllm_check_stops_only_a_vllm_run_without_vllm(tmp_path, backend, found, rc):
    vllm_py = vllm_python(tmp_path) if found else str(tmp_path / "missing")
    out = subprocess.run(["bash", "-c", f"source {LIB}; vllm_check {sys.executable}"],
                         env={**os.environ, "GEN_BACKEND": backend, "VLLM_PY": vllm_py},
                         capture_output=True, text=True)
    assert out.returncode == rc, out.stderr
    if rc:
        assert "no vLLM environment" in out.stderr


@pytest.mark.parametrize("runner,args", [
    ("scripts/qwen/ced/run_cllora.sh", ["--method", "inclora", "--data-root", "data/tacred_perm0", "--num-tasks", "10"]),
    ("scripts/qwen/ced/run_ced_v2.sh", ["--run-name", "vllm_refusal_probe", "--gpus", "0"]),
])
def test_runners_refuse_vllm_without_an_environment_before_any_run_directory(runner, args, tmp_path):
    save = tmp_path / "run"
    extra = ["--py", sys.executable, "--save", str(save)] if "cllora" in runner else []
    env = {**os.environ, "GEN_BACKEND": "vllm", "VLLM_PY": str(tmp_path / "missing"),
           "ENV_BIN": os.path.dirname(sys.executable)}
    out = subprocess.run(["bash", runner] + args + extra, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode != 0
    assert "no vLLM environment" in out.stdout + out.stderr
    assert not save.exists() and not os.path.exists("results/qwen3/ced/vllm_refusal_probe")


def test_the_ced_runner_records_the_backend_in_its_manifest():
    script = open("scripts/qwen/ced/run_ced_v2.sh").read()
    manifest_line = next(line for line in script.splitlines() if line.startswith("MANIFEST_CONFIG="))
    assert ";gen=${GEN_BACKEND};" in manifest_line


def test_the_runners_hand_the_backend_on():
    ced = open("scripts/qwen/ced/run_ced_v2.sh").read()
    assert 'OPTS+=" --gen-backend ${GEN_BACKEND}"' in ced and 'PL_OPTS+=" --gen-backend ${GEN_BACKEND}"' in ced
    assert '--gen-backend "${GEN_BACKEND:-hf}"' in open("scripts/qwen/ced/run_cllora.sh").read()


def test_the_ced_runner_passes_gradient_checkpointing_on():
    script = open("scripts/qwen/ced/run_ced_v2.sh").read()
    assert "GRAD_CKPT=${GRAD_CKPT:-0}" in script and "--grad-ckpt) GRAD_CKPT=$2; shift 2;;" in script
    assert '[ "${GRAD_CKPT}" = "1" ] && OPTS+=" --gradient-checkpointing"' in script


def test_the_ced_runner_records_compiled_generation_in_its_manifest():
    script = open("scripts/qwen/ced/run_ced_v2.sh").read()
    manifest_line = next(line for line in script.splitlines() if line.startswith("MANIFEST_CONFIG="))
    assert ";cgen=${COMPILE_GEN};" in manifest_line
    assert '[ "${COMPILE_GEN}" = "1" ] && OPTS+=" --compile-generation"' in script
