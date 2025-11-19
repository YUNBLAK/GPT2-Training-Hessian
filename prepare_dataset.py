"""
Tokenize local Parquet files (e.g., dataset/SAMPLE10B/*.parquet) and write uint16 token shards (.npy).
No Hugging Face download. Uses GPT-2 (tiktoken) tokenizer.

Usage:
  python prepare_dataset_local.py \
    --input_dir dataset/SAMPLE10B \
    --pattern "*.parquet" \
    --output_dir /project2/karimire_1756/daniel/GPT2/data/edu_fineweb10B \
    --shard_size 10000000
"""

import os
import glob
import argparse
import numpy as np
import multiprocessing as mp
import tiktoken
import pyarrow.parquet as pq
from tqdm import tqdm

# ----------------- Args -----------------
parser = argparse.ArgumentParser(description="Local Parquet → token shards (.npy)")
parser.add_argument("--input_dir", type=str, default="dataset/SAMPLE10B",
                    help="Directory containing local parquet files")
parser.add_argument("--pattern", type=str, default="*.parquet",
                    help="Glob pattern for parquet files")
parser.add_argument("--output_dir", type=str, default="data/edu_fineweb10B",
                    help="Directory to save tokenized .npy shards")
parser.add_argument("--shard_size", type=int, default=int(1e8),
                    help="Tokens per shard (default: 1e8)")
parser.add_argument("--pa_batch_rows", type=int, default=2048,
                    help="Parquet batch rows per read (memory control)")
parser.add_argument("--mp_buffer", type=int, default=8192,
                    help="Texts per multiprocessing job (tokenize batch size)")
args = parser.parse_args()

INPUT_DIR = os.path.abspath(args.input_dir)
OUTPUT_DIR = os.path.abspath(args.output_dir)
PATTERN = args.pattern
SHARD_SIZE = int(args.shard_size)
PA_BATCH_ROWS = int(args.pa_batch_rows)
MP_BUFFER = int(args.mp_buffer)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Input dir: {INPUT_DIR}")
print(f"[INFO] Pattern:   {PATTERN}")
print(f"[INFO] Output dir: {OUTPUT_DIR}")
print(f"[INFO] Shard size: {SHARD_SIZE:,} tokens")

# ----------------- Tokenizer -----------------
enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]

def tokenize_text(text: str) -> np.ndarray:
    toks = [EOT]
    toks.extend(enc.encode_ordinary(text))
    return np.asarray(toks, dtype=np.uint16)

def tokenize_batch(texts):
    return [tokenize_text(t) for t in texts]

# ----------------- Parquet iterators -----------------
def iter_text_rows(parquet_path, batch_rows=PA_BATCH_ROWS):
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=["text"], batch_size=batch_rows):
        col = batch.column(0)
        for txt in col.to_pylist():
            if txt is None:
                continue
            yield txt

def yield_text_batches(files, buf_cap=MP_BUFFER):
    buf = []
    for f in files:
        for t in iter_text_rows(f, batch_rows=PA_BATCH_ROWS):
            buf.append(t)
            if len(buf) >= buf_cap:
                yield buf
                buf = []
    if buf:
        yield buf

# ----------------- Shard writer -----------------
def write_full_shard(path, arr_uint16):
    np.save(path, arr_uint16)

def write_partial_shard(path, arr_uint16, valid_len):
    np.save(path, arr_uint16[:valid_len])

# ----------------- Main -----------------
def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, PATTERN)))
    if not files:
        raise FileNotFoundError(f"No parquet files matching {PATTERN} in {INPUT_DIR}")

    shard_idx = 0
    token_count = 0
    all_tokens = np.empty((SHARD_SIZE,), dtype=np.uint16)
    pbar = None
    first_full_shard_written = False

    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        for token_lists in pool.imap(tokenize_batch, yield_text_batches(files), chunksize=1):
            for toks in token_lists:
                tlen = len(toks)
                if token_count + tlen < SHARD_SIZE:
                    all_tokens[token_count:token_count + tlen] = toks
                    token_count += tlen
                    if pbar is None:
                        pbar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"shard {shard_idx}")
                    pbar.update(tlen)
                else:
                    split = "val" if shard_idx == 0 else "train"
                    shard_path = os.path.join(OUTPUT_DIR, f"edufineweb_{split}_{shard_idx:06d}.npy")
                    remainder = SHARD_SIZE - token_count
                    if pbar is None:
                        pbar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"shard {shard_idx}")
                    pbar.update(remainder)
                    all_tokens[token_count:token_count + remainder] = toks[:remainder]
                    write_full_shard(shard_path, all_tokens)
                    first_full_shard_written = True
                    shard_idx += 1
                    pbar.close(); pbar = None

                    leftover = tlen - remainder
                    all_tokens[0:leftover] = toks[remainder:]
                    token_count = leftover

    # Save final partial shard
    if token_count > 0:
        # 최소 1개 train 샤드 보장
        split = "train" if not first_full_shard_written else ("train" if shard_idx > 0 else "val")
        shard_path = os.path.join(OUTPUT_DIR, f"edufineweb_{split}_{shard_idx:06d}.npy")
        write_partial_shard(shard_path, all_tokens, token_count)
        print(f"[INFO] Saved final shard: {shard_path} ({token_count:,} tokens)")
    if pbar is not None:
        pbar.close()

    print("[INFO] ✅ Tokenization and sharding complete.")

if __name__ == "__main__":
    main()
