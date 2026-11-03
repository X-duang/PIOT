from argparse import Namespace
from pathlib import Path
from analysis.prior_shift import _planned_run_dir, _fixed_sample_size, resample_target
from main.run_identity import build_run_dir

def test_auto_planned_run_matches_training_identity(tmp_path):
    target = tmp_path / "target.csv"
    target.write_text("pairs,labels\na,1\nb,0\n", encoding="utf-8")
    a = Namespace(src="cameras", tgt="computers", tau=1.0, ot_reg=0.1,
        ot_normalize=True, batch_size=32, eval_protocol="inductive",
        max_seq_length=128, num_epochs=40, ot_max_iter=300, ot_tol=1e-7,
        analysis_size=256, analysis_seed=0, deterministic=False,
        alignment_weight=0.01, lr=1.5e-5, results_root=str(tmp_path))
    run = _planned_run_dir(a, str(target), 0.05, 0, 0, "ot_auto")
    actual = Namespace(experiment="experiment_B", tier="tier_3", task="cameras_to_computers",
        src=a.src, tgt=a.tgt, target_file=str(target), method="ot_auto", seed=0, train_seed=0,
        alignment_weight=a.alignment_weight, tau=a.tau, ot_reg=a.ot_reg, cost="l2",
        cost_mode="class_blind", batch_size=a.batch_size, lr=a.lr, ot_normalize=True,
        eval_protocol=a.eval_protocol, run_tag="prior0.05_rep0", max_seq_length=128,
        num_epochs=40, ot_max_iter=300, ot_tol=1e-7, analysis_size=256, analysis_seed=0,
        deterministic=False, results_root=str(tmp_path))
    assert run == build_run_dir(actual)

def test_fixed_size_covers_all_prior_levels():
    size = _fixed_sample_size([1]*30, [0]*70, [0.05,0.10,0.17,0.27,0.40,0.55], 100)
    assert all(round(p*size) <= 30 and size-round(p*size) <= 70
               for p in [0.05,0.10,0.17,0.27,0.40,0.55])
