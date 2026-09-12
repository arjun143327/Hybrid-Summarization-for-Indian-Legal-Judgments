"""
download_w2v_google_news.py — Download word2vec-google-news-300 and convert to memory-mapped .kv format.
Tracks download progress, memory consumption, and load time.
"""

import os
import sys
import time
import urllib.request
import psutil
from gensim.models import KeyedVectors

URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-google-news-300/word2vec-google-news-300.gz"
TARGET_DIR = os.path.join(os.path.expanduser("~"), "gensim-data", "word2vec-google-news-300")
GZ_FILE = os.path.join(TARGET_DIR, "word2vec-google-news-300.gz")
KV_FILE = os.path.join(TARGET_DIR, "vectors.kv")


def get_memory_info():
    proc = psutil.Process()
    vm = psutil.virtual_memory()
    return {
        "proc_rss_mb": proc.memory_info().rss / (1024 * 1024),
        "system_avail_mb": vm.available / (1024 * 1024),
        "system_total_mb": vm.total / (1024 * 1024),
    }


def download_file(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"

    if os.path.exists(dest_path):
        expected_size = 1743563840
        actual_size = os.path.getsize(dest_path)
        if actual_size == expected_size:
            print(f"Archive already downloaded at {dest_path} ({actual_size:,} bytes).")
            return
        else:
            print(f"Existing file incomplete ({actual_size:,} bytes vs {expected_size:,} expected). Re-downloading...")

    print(f"Downloading from {url} to {dest_path}...")
    start_time = time.time()
    last_log_time = start_time

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response, open(temp_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 4 * 1024 * 1024  # 4MB chunks

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_log_time >= 5.0 or downloaded == total_size:
                elapsed = now - start_time
                pct = (downloaded / total_size * 100) if total_size else 0
                speed_mb = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                print(f"  Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) at {speed_mb:.2f} MB/s", flush=True)
                last_log_time = now

    os.replace(temp_path, dest_path)
    total_time = time.time() - start_time
    print(f"Download complete in {total_time:.2f}s ({os.path.getsize(dest_path):,} bytes).", flush=True)


def convert_and_verify():
    mem_before = get_memory_info()
    print(f"Pre-load RAM: Process RSS = {mem_before['proc_rss_mb']:.1f} MB, Available System RAM = {mem_before['system_avail_mb']:.1f} MB", flush=True)

    if os.path.exists(KV_FILE):
        print(f"Memory-mapped binary vectors.kv already exists at {KV_FILE}.", flush=True)
    else:
        print("Decompressing and loading word2vec-google-news-300 from .gz format...", flush=True)
        t0 = time.time()
        wv = KeyedVectors.load_word2vec_format(GZ_FILE, binary=True)
        load_time = time.time() - t0
        mem_loaded = get_memory_info()
        print(f"Loaded {len(wv):,} vectors of dimension {wv.vector_size} in {load_time:.2f}s.", flush=True)
        print(f"Post-load RAM: Process RSS = {mem_loaded['proc_rss_mb']:.1f} MB, Available System RAM = {mem_loaded['system_avail_mb']:.1f} MB", flush=True)

        print(f"Saving to memory-mapped native KeyedVectors format at {KV_FILE}...", flush=True)
        t0 = time.time()
        wv.save(KV_FILE)
        save_time = time.time() - t0
        print(f"Saved vectors.kv in {save_time:.2f}s.", flush=True)
        del wv

    # Verify mmap loading speed and memory footprint
    print("\nVerifying memory-mapped load speed with mmap='r'...", flush=True)
    t0 = time.time()
    wv_mmap = KeyedVectors.load(KV_FILE, mmap="r")
    mmap_load_time = time.time() - t0
    mem_mmap = get_memory_info()
    print(f"mmap load time: {mmap_load_time:.4f}s!", flush=True)
    print(f"mmap Process RSS: {mem_mmap['proc_rss_mb']:.1f} MB, Available System RAM: {mem_mmap['system_avail_mb']:.1f} MB", flush=True)

    # Quick sanity vector lookup
    test_word = "court"
    print(f"Vector lookup for '{test_word}': shape={wv_mmap[test_word].shape}, norm={float(wv_mmap.get_vector(test_word, norm=True).sum()):.4f}", flush=True)
    print("Verification SUCCESSFUL!", flush=True)


if __name__ == "__main__":
    download_file(URL, GZ_FILE)
    convert_and_verify()
