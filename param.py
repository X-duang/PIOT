# params for dataset and data loader
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
data_root = "data"

# target split fractions (final protocol): pool/valid/test = 70/10/20.
# Single source of truth: consumed by main/experiment_common.py (splitting)
# and main/run_identity.py (run identity), so changing them re-hashes runs.
target_valid_frac = 0.10
target_test_frac = 0.20  # inductive protocol only; fraction of the full target set

# Stratified target split (final protocol). ER 候选对类别严重失衡
# （正例率 9%~32%），label-blind 随机划分会让小目标域（fz/b2/ri）的
# valid/test 正例数大幅偏离全集比例，污染模型选择与最终 F1。
# 分层仅用目标标签【构造划分】，保证 pool/valid/test 正例率与全集一致；
# 适配批仍然无标签（include_labels=False），标签防火墙不受影响。
# 纳入 run identity：开启后所有 run 重新哈希（划分本身变了，理应重跑）。
target_split_stratified = True

# params for source dataset
src_encoder_path = "source-encoder.pt"
src_classifier_path = "source-classifier.pt"

# params for target dataset
tgt_encoder_path = "target-encoder.pt"
tgt_classifier_path = "target-classifier.pt"
model_root = _os.path.join(_ROOT, "checkpoint")
d_model_path = "critic.pt"
mlp_model_path = "mlp.pt"

# params for training network
c_learning_rate = 2e-5
d_learning_rate = 1.5e-5
weight_decay = 1e-6
hidden_size = 768
intermediate_size = 3072
kernel_num = 20
kernel_sizes = [3, 4, 5]
pretrain = True
dropout = 0.1
num_labels = 2
h_dim = 300

"""Special tokens."""
PAD = 0
UNK = 1
SOS = 2
EOS = 3
