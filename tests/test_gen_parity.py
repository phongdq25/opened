"""tools/gen_parity.py's helpers (the GPU runs are Task 7's checks)."""
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location("gen_parity", os.path.join("tools", "gen_parity.py"))
gp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gp)


def test_the_last_test_line_of_a_log_gives_the_metrics(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("dev | avg_loss: 0.1 | {}\n"
                   "test | avg_loss: 0.2 | {'trigger': {'f1': 0.5}}\n"
                   "test | avg_loss: 0.3 | {'exact_match': 1.0, 'trigger': {'precision': 0.7, 'recall': 0.7, 'f1': 0.7}}\n")
    assert gp.last_test_metrics(str(log))["trigger"]["f1"] == 0.7


def test_the_evaluation_arguments_parse_like_the_runners(tmp_path):
    from argparse import Namespace
    args = gp.eval_args(Namespace(base="models/Qwen3-0.6B", data_dir="processed_data/x/0/qwen/", out=str(tmp_path),
                                  strict=True))
    assert (args.eval_gen, args.strict_generation, args.top_p, args.temperature, args.top_k) == (True, True, 0.95, 0.5, 0)
    assert (args.max_length, args.max_prompt_length, args.eval_batch_size) == (768, 460, 128)
    assert args.save.startswith(str(tmp_path))


def test_the_summary_compares_the_means_in_points():
    rows = [{"backend": "hf", "f1": 0.60}, {"backend": "hf", "f1": 0.62},
            {"backend": "vllm", "f1": 0.605}, {"backend": "vllm", "f1": 0.615}]
    assert gp.summary(rows) == {"hf": {"mean": 61.0, "spread": 2.0}, "vllm": {"mean": 61.0, "spread": 1.0},
                                "gap": 0.0}
