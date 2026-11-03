# Dataset preparation

The 13 paper CSVs match the [DADER repository](https://github.com/ruc-datalab/DADER) at commit `80f3445` after normalizing CRLF to LF. Run `python scripts/download_datasets.py` from the repository root to download them and verify SHA-256 digests in `checksums.csv`. Files are placed under `data/<dataset>/<dataset>.csv`.

These datasets originate from multiple third-party collections, including DeepMatcher and WDC. Their source and redistribution terms remain separate from the license for PIVOT code. See [DeepMatcher datasets](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) and the [WDC product corpus](https://webdatacommons.org/largescaleproductcorpus/). The dataset files are fetched from their existing public upstream and are not included in this repository.
