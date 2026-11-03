# -*- coding: utf-8 -*-
"""Runner and identity tests for the faithful DeepJDOT experiment path."""
from argparse import Namespace

from main.run_identity import build_run_dir
from run_experiments import _command, _method_specs, _resolved_config, _task_specs


def _args(method, **overrides):
    values = dict(
        method=method, task="T1", src="wa1", tgt="ab", train_seed=0,
        seed=42, num_epochs=20, run_tag="audit", results_root="results",
        deepjdot_alpha=0.001, deepjdot_lambda_t=0.0001,
        deepjdot_solver="emd",
    )
    values.update(overrides)
    return Namespace(**values)


def test_faithful_runner_forwards_original_parameters():
    manifest_path = "manifests/experiment_A_deepjdot_faithful_ep20.json"
    from run_experiments import load_manifest
    manifest = load_manifest(manifest_path)
    method = _method_specs(manifest, ["deepjdot_faithful"])[0]
    task = _task_specs(manifest, ["T1"])[0]
    config = _resolved_config(manifest, task, method, 42)
    config["results_root"] = "results/results_deepjdot_faithful_ep20"
    command = _command(config, method)
    assert "--method" in command
    assert command[command.index("--method") + 1] == "deepjdot_faithful"
    assert command[command.index("--num_epochs") + 1] == "20"
    assert command[command.index("--selection") + 1] == "last_epoch"
    assert command[command.index("--deepjdot_alpha") + 1] == "0.001"
    assert command[command.index("--deepjdot_lambda_t") + 1] == "0.0001"
    assert command[command.index("--deepjdot_solver") + 1] == "emd"


def test_faithful_identity_changes_only_for_faithful_parameters():
    base = _args("deepjdot_faithful")
    changed = _args("deepjdot_faithful", deepjdot_lambda_t=0.001)
    assert build_run_dir(base) != build_run_dir(changed)



def test_faithful_manifest_has_twelve_tasks_and_one_method():
    from run_experiments import load_manifest
    for epochs in (20, 40):
        manifest = load_manifest(f"manifests/experiment_A_deepjdot_faithful_ep{epochs}.json")
        assert len(manifest["tasks"]) == 12
        assert len(manifest["methods"]) == 1
        assert manifest["defaults"]["num_epochs"] == epochs
        assert manifest["defaults"]["deepjdot_solver"] == "emd"
