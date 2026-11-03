# train/run_logger.py
"""统一的训练记录器（v2）。

相对旧版的变化：
  - final runtime 不记录旧动态采样、mass allocation 或 coreset 决策；
  - 每 epoch 只记 valid 指标（评估削减）；训练期不再每 epoch 评大池；
  - 新增 OT 诊断字段（transported_mass 等，支撑论文 RQ4 分析）；
  - 新增 final 成绩：best checkpoint 在全量目标池上的 P/R/F1；
  - 崩溃安全：异常也会 flush 并写 status=interrupted 的 summary。

用法:
    logger = RunLogger(args, method="ot_balanced")
    logger.log_config(extra={...})
    logger.log_step(epoch, step, total_steps, cls_val, ot_val)
    logger.log_epoch(epoch, f1_valid, eloss_valid, ot_diag=diag)
    logger.log_final(f1, precision, recall)
    logger.finalize()
"""
import os
import csv
import json
import time

CSV_COLUMNS = ["timestamp", "record_type", "epoch", "step",
               "cls_loss", "ot_loss", "total_mass",
               "f1_valid", "eval_loss_valid"]


class RunLogger:
    def __init__(self, args, method="ot", result_dir=None):
        self.args = args
        self.method = method
        self.result_dir = result_dir or getattr(args, "result_dir", "../result")
        os.makedirs(self.result_dir, exist_ok=True)

        stem = (
            f"{getattr(args, 'task', args.src + '_to_' + args.tgt)}"
            f"_train{getattr(args, 'train_seed', 'NA')}"
            f"_split{getattr(args, 'seed', 'NA')}_{method}"
        )
        run_tag = getattr(args, "run_tag", "")
        if run_tag:
            stem += f"_{run_tag}"
        self.csv_path = os.path.join(self.result_dir, stem + "_log.csv")
        self.json_path = os.path.join(self.result_dir, stem + "_summary.json")

        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(CSV_COLUMNS)
        self._csv_file.flush()

        self.config = {}
        self.history = []
        self.best_valid_f1 = -1.0  # -1 起步：首个 epoch 必记为 best
        self.best_epoch = -1
        self.final = None
        self.status = "running"
        print(f"[RunLogger] CSV -> {self.csv_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.status = "completed" if exc_type is None else f"interrupted:{exc_type.__name__}"
        self._write_summary()
        try:
            self._csv_file.close()
        except Exception:
            pass
        return False  # 不吞异常

    def _row(self, record_type, epoch="", step="", cls_loss="", ot_loss="",
             total_mass="", f1_valid="", el_valid=""):
        self._writer.writerow([int(time.time()), record_type, epoch, step,
                               cls_loss, ot_loss, total_mass, f1_valid, el_valid])
        self._csv_file.flush()

    def log_config(self, extra=None):
        """Persist the frozen-paper run configuration."""
        self.config = dict(extra or {})

    def log_step(self, epoch, step, total_steps, cls_val, ot_val, total_mass=""):
        self._row("step", epoch=epoch + 1, step=f"{step + 1}/{total_steps}",
                  cls_loss=cls_val, ot_loss=ot_val, total_mass=total_mass)

    def log_epoch(self, epoch, f1_valid, eloss_valid=None, ot_diag=None):
        self._row("epoch_end", epoch=epoch + 1,
                  f1_valid=f1_valid,
                  el_valid="" if eloss_valid is None else eloss_valid)
        self.history.append({"epoch": epoch + 1, "f1_valid": f1_valid,
                             **(ot_diag or {})})
        is_new_best = f1_valid > self.best_valid_f1
        if is_new_best:
            self.best_valid_f1 = f1_valid
            self.best_epoch = epoch + 1
        # 增量写 summary：中途崩溃也保留到上一轮为止的完整记录
        self._write_summary()
        return is_new_best

    def log_final(self, f1, precision, recall, extra=None):
        """best checkpoint 在全量目标池（或 held-out test）上的最终成绩。"""
        self.final = {"final_f1": f1, "final_precision": precision,
                      "final_recall": recall, **(extra or {})}
        self._write_summary()

    def finalize(self):
        self.status = "completed"
        self._write_summary()
        try:
            self._csv_file.close()
        except Exception:
            pass

    def _write_summary(self):
        args = self.args
        summary = {
            "src": getattr(args, "src", None),
            "tgt": getattr(args, "tgt", None),
            "method": self.method,
            "status": self.status,
            "seed": getattr(args, "train_seed", None),
            "num_epochs": getattr(args, "num_epochs", None),
            "batch_size": getattr(args, "batch_size", None),
            "model": getattr(args, "model", None),
            "experiment": getattr(args, "experiment", None),
            "tier": getattr(args, "tier", None),
            "task": getattr(args, "task", None),
            "eval_protocol": getattr(args, "eval_protocol", None),
            "result_dir": self.result_dir,
            "config": self.config,
            "best_valid_f1": self.best_valid_f1,
            "best_epoch": self.best_epoch,
            "final": self.final,
            "history": self.history,
        }
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[RunLogger] 写 summary 失败: {e}")
