import csv
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import param


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids=None, input_mask=None, segment_ids=None,label_id=None,exm_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.exm_id = exm_id
class InputFeaturesED(object):
    """A single set of features of data for ED."""
    def __init__(self, input_ids, attention_mask,label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_id = label_id

def CSV2Array(path):
    """Read data from csv"""
    data = pd.read_csv(path, encoding='latin')
    pairs, labels = data.pairs.values.tolist(), data.labels.values.tolist()
    return pairs, labels

def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(args, net, restore=None):
    """ restore model weights """
    if restore is not None:
        checkpoint_dir = getattr(args, "checkpoint_dir", None)
        if checkpoint_dir:
            path = os.path.join(checkpoint_dir, restore)
        else:
            path = os.path.join(param.model_root, args.src, args.model,
                                str(args.train_seed), restore)
        if os.path.exists(path):
            net.load_state_dict(torch.load(path, map_location="cpu"))
            print("Restore model from: {}".format(os.path.abspath(path)))

    """ check if cuda is available """
    if torch.cuda.is_available():
        if getattr(args, "deterministic", False):
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True
        net.cuda()
    return net

def save_model(args, net, name):
    """Save trained model."""
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    folder = checkpoint_dir or os.path.join(
        param.model_root, args.src, args.model, str(args.train_seed)
    )
    path = os.path.join(folder, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), path)
    print("save pretrained model to: {}".format(path))


def convert_examples_to_features(pairs, labels, max_seq_length, tokenizer,
                                 cls_token='[CLS]', sep_token='[SEP]',
                                 pad_token=0,csv_writer=None,exp_idx=-1):
    features = []
    for ex_index, (pair, label) in enumerate(zip(pairs, labels)):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(pairs)))
        # add ER situation
        if sep_token in pair:
            left = pair.split(sep_token)[0]
            right = pair.split(sep_token)[1]
            ltokens = tokenizer.tokenize(left)
            rtokens = tokenizer.tokenize(right)
            more = len(ltokens) + len(rtokens) - max_seq_length + 3
            if more > 0:
                if more <len(rtokens) : # remove excessively long string
                    rtokens = rtokens[:(len(rtokens) - more)]
                elif more <len(ltokens):
                    ltokens = ltokens[:(len(ltokens) - more)]
                else:
                    print("too long!")
                    continue
            tokens = [cls_token] + ltokens + [sep_token] + rtokens + [sep_token]
            segment_ids = [0]*(len(ltokens)+2) + [1]*(len(rtokens)+1)
        else:
            tokens = tokenizer.tokenize(pair)
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
            tokens = [cls_token] + tokens + [sep_token]
            segment_ids = [0]*(len(tokens))
        if ex_index == exp_idx:
            """This is for recording attention"""
            with open('tokens.csv', 'w', newline='') as csvfile:
                writer  = csv.writer(csvfile)
                writer.writerow(tokens)
            with open('token_type_ids.csv', 'w', newline='') as csvfile:
                writer  = csv.writer(csvfile)
                writer.writerow(segment_ids)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids = segment_ids,
                          label_id=label,
                          exm_id=ex_index))
        if csv_writer != None:
            """Record training data for semi"""
            csv_writer.writerow([ex_index, pair, label])
    return features

def get_data_loader(features, batch_size,flag):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_exm_ids = torch.tensor([f.exm_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask,all_segment_ids, all_label_ids,all_exm_ids)
    sampler = RandomSampler(dataset)
    if flag == "dev":
        """Read all data"""
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    else:
        """Delet the last incomplete epoch"""
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    return dataloader

def get_data_loaderED(features, batch_size,flag):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask,all_label_ids)
    sampler = RandomSampler(dataset)
    if flag == "dev":
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    else:
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    return dataloader


def bart_convert_examples_to_features(pairs, labels, max_seq_length, tokenizer, pad_token=0, cls_token='<s>',sep_token='</s>'):
    features = []
    for ex_index, (pair, label) in enumerate(zip(pairs, labels)):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(pairs)))
        if sep_token in pair:
            left = pair.split(sep_token)[0]
            right = pair.split(sep_token)[1]
            ltokens = tokenizer.tokenize(left)
            rtokens = tokenizer.tokenize(right)
            more = len(ltokens) + len(rtokens) - max_seq_length + 3
            if more > 0:
                if more <len(rtokens) : #从rtokens中删除多余的部分
                    rtokens = rtokens[:(len(rtokens) - more)]
                elif more <len(ltokens):
                    ltokens = ltokens[:(len(ltokens) - more)]
                else:
                    print("bad example!")
                    continue
            tokens =  [cls_token] +ltokens + [sep_token] + rtokens + [sep_token]
            segment_ids = [0]*(len(ltokens)+2) + [1]*(len(rtokens)+1)
        else:
            tokens = tokenizer.tokenize(pair)
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
            tokens = [cls_token] + tokens + [sep_token]
            segment_ids = [0]*(len(tokens))
            
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        features.append(InputFeaturesED(input_ids=input_ids,
                        attention_mask=input_mask,
                        label_id=label
                        ))
    return features
  


def MMD(source, target):
    """Compute MMD"""
    mmd_loss = torch.exp(-1 / (source.mean(dim=0) - target.mean(dim=0)).norm())
    return mmd_loss


# =====================================================================
# v2 新增：批量 tokenization + 特征磁盘缓存 + GPU 驻留批迭代
# （见《代码优化方案》§6：消除每 run 重复 tokenize 与 per-batch H2D 传输）
# =====================================================================
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FEATURE_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "_cache")


def convert_examples_to_features_fast(pairs, labels, max_seq_length, tokenizer,
                                      cls_token='[CLS]', sep_token='[SEP]',
                                      pad_token=0, chunk_size=4096):
    """convert_examples_to_features 的批量版，输出与原版逐位一致。

    原版逐样本 tokenizer.tokenize + convert_tokens_to_ids，每 run 重复执行；
    本版用批量 tokenizer(list, add_special_tokens=False) 直接拿 input_ids，
    截断/拼接规则与原版完全相同（[CLS] left [SEP] right [SEP]，超长先截右再截左，
    左右都放不下则丢弃该样本）。
    """
    # 1) 按 [SEP] 拆左右实体
    lefts, rights = [], []
    for pair in pairs:
        if sep_token in pair:
            left, right = pair.split(sep_token)[0], pair.split(sep_token)[1]
        else:
            left, right = pair, ""
        lefts.append(left)
        rights.append(right)

    # 2) 批量分词（直接得 id 序列）
    lid_list, rid_list = [], []
    for i in range(0, len(lefts), chunk_size):
        enc_l = tokenizer(lefts[i:i + chunk_size], add_special_tokens=False)["input_ids"]
        enc_r = tokenizer(rights[i:i + chunk_size], add_special_tokens=False)["input_ids"]
        lid_list.extend(enc_l)
        rid_list.extend(enc_r)

    cls_id = tokenizer.convert_tokens_to_ids(cls_token)
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)

    # 3) 按原版规则组装
    features = []
    for ex_index, (lids, rids, label) in enumerate(zip(lid_list, rid_list, labels)):
        if rights[ex_index] != "" or sep_token in pairs[ex_index]:
            more = len(lids) + len(rids) - max_seq_length + 3
            if more > 0:
                if more < len(rids):
                    rids = rids[:len(rids) - more]
                elif more < len(lids):
                    lids = lids[:len(lids) - more]
                else:
                    continue  # 原版 "too long!" 丢弃
            input_ids = [cls_id] + lids + [sep_id] + rids + [sep_id]
            segment_ids = [0] * (len(lids) + 2) + [1] * (len(rids) + 1)
        else:
            lids = lids[:max_seq_length - 2]
            input_ids = [cls_id] + lids + [sep_id]
            segment_ids = [0] * len(input_ids)

        input_mask = [1] * len(input_ids)
        pad_len = max_seq_length - len(input_ids)
        input_ids += [pad_token] * pad_len
        input_mask += [0] * pad_len
        segment_ids += [0] * pad_len

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label,
                                      exm_id=ex_index))
    return features


def load_or_build_features(tag, pairs, labels, max_seq_length, tokenizer,
                           cache_dir=FEATURE_CACHE_DIR, use_cache=True):
    """特征磁盘缓存：同一 (数据集, seq_len, tokenizer) 只 tokenize 一次。

    缓存 key 含 tokenizer 名与 seq_len 的哈希，防配置漂移读旧缓存。
    """
    key = hashlib.md5(
        f"{tag}|{max_seq_length}|{tokenizer.name_or_path}".encode()).hexdigest()[:8]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{tag}_seq{max_seq_length}_{key}.pt")

    if use_cache and os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=True)
        if blob["n"] == len(labels):
            print(f"[cache] 命中 {cache_path}")
            return _blob_to_features(blob)
        print(f"[cache] 样本数不一致，重建 {cache_path}")

    features = convert_examples_to_features_fast(pairs, labels, max_seq_length, tokenizer)
    blob = _features_to_blob(features)
    torch.save(blob, cache_path)
    print(f"[cache] 写入 {cache_path}")
    return features


def _features_to_blob(features):
    return {
        "n": len(features),
        "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
        "input_mask": torch.tensor([f.input_mask for f in features], dtype=torch.long),
        "segment_ids": torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        "label_id": torch.tensor([f.label_id for f in features], dtype=torch.long),
        "exm_id": torch.tensor([f.exm_id for f in features], dtype=torch.long),
    }


def _blob_to_features(blob):
    features = []
    for i in range(blob["n"]):
        features.append(InputFeatures(
            input_ids=blob["input_ids"][i].tolist(),
            input_mask=blob["input_mask"][i].tolist(),
            segment_ids=blob["segment_ids"][i].tolist(),
            label_id=int(blob["label_id"][i]),
            exm_id=int(blob["exm_id"][i])))
    return features


class GpuBatches:
    """GPU 驻留批迭代器：数据集一次性上卡，每 epoch randperm 切片取批。

    ER 候选对数据集很小（最大 ~29k × 128 × int64 ≈ 29MB/tensor），
    全量驻留 GPU 可消除全部 per-batch H2D 传输与 DataLoader 开销。
    yield 的五元组与原 get_data_loader 完全一致（含 exm_id）。
    """

    def __init__(self, features, batch_size, shuffle=True, drop_last=False,
                 device=None, include_labels=True, seed=None):
        blob = _features_to_blob(features)
        self.device = device or (torch.device("cuda")
                                 if torch.cuda.is_available() else torch.device("cpu"))
        self.input_ids = blob["input_ids"].to(self.device)
        self.input_mask = blob["input_mask"].to(self.device)
        self.segment_ids = blob["segment_ids"].to(self.device)
        self.labels = (
            blob["label_id"].to(self.device) if include_labels else None
        )
        self.exm_ids = blob["exm_id"].to(self.device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.include_labels = include_labels
        self.seed = None if seed is None else int(seed)
        self._generator = None
        if self.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)

    @property
    def n_samples(self):
        return self.input_ids.size(0)

    def __len__(self):
        n, bs = self.n_samples, self.batch_size
        return n // bs if self.drop_last else (n + bs - 1) // bs

    def __iter__(self):
        n, bs = self.n_samples, self.batch_size
        if self.shuffle:
            if self._generator is None:
                perm = torch.randperm(n, device=self.device)
            else:
                perm = torch.randperm(n, generator=self._generator)
                perm = perm.to(self.device)
        else:
            perm = torch.arange(n, device=self.device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if self.drop_last and idx.numel() < bs:
                break
            if self.include_labels:
                yield (self.input_ids[idx], self.input_mask[idx],
                       self.segment_ids[idx], self.labels[idx],
                       self.exm_ids[idx])
            else:
                yield (self.input_ids[idx], self.input_mask[idx],
                       self.segment_ids[idx], self.exm_ids[idx])

    def subset(self, indices):
        """按索引取子集（coreset 用），返回新的 GpuBatches。"""
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        new = object.__new__(GpuBatches)
        new.device = self.device
        new.input_ids = self.input_ids[idx]
        new.input_mask = self.input_mask[idx]
        new.segment_ids = self.segment_ids[idx]
        new.labels = self.labels[idx] if self.labels is not None else None
        new.exm_ids = self.exm_ids[idx]
        new.batch_size = self.batch_size
        new.shuffle = self.shuffle
        new.drop_last = self.drop_last
        new.include_labels = self.include_labels
        new.seed = self.seed
        new._generator = None
        if new.seed is not None:
            new._generator = torch.Generator()
            new._generator.manual_seed(new.seed)
        return new
