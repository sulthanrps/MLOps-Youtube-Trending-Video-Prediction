import argparse
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8090/invocations")
    ap.add_argument("-n", "--requests", type=int, default=300)
    ap.add_argument("-c", "--concurrency", type=int, default=20)
    a = ap.parse_args()

    payload = (ROOT / "sample_serving_request.json").read_text()
    headers = {"Content-Type": "application/json"}
    session = requests.Session()

    def call(_):
        t0 = time.perf_counter()
        try:
            r = session.post(a.url, data=payload, headers=headers, timeout=30)
            ms = (time.perf_counter() - t0) * 1000
            preds = json.dumps(r.json().get("predictions")) if r.ok else None
            return r.status_code, r.headers.get("X-Upstream-Addr", "-"), ms, preds
        except requests.RequestException as e:
            return f"ERR {type(e).__name__}", "-", (time.perf_counter() - t0) * 1000, None

    print(f"Mengirim {a.requests} request ({a.concurrency} paralel) ke {a.url} ...")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        results = list(ex.map(call, range(a.requests)))
    elapsed = time.perf_counter() - start

    ok = [r for r in results if r[0] == 200]
    lat = sorted(r[2] for r in ok)
    per_replica = Counter(r[1] for r in ok)
    outputs = Counter(r[3] for r in ok)

    print(f"\nSelesai dalam {elapsed:.2f} s | throughput {len(results) / elapsed:.1f} req/s")
    print(f"Sukses (200): {len(ok)}/{len(results)} | gagal: {len(results) - len(ok)}")
    if lat:
        p95 = lat[max(0, int(len(lat) * 0.95) - 1)]
        print(f"Latensi: p50 {statistics.median(lat):.1f} ms | p95 {p95:.1f} ms | maks {lat[-1]:.1f} ms")
    print(f"\nSebaran request per replika ({len(per_replica)} replika menjawab):")
    for addr, n in sorted(per_replica.items()):
        print(f"  {addr:<22} {n:>5} request  {'#' * max(1, round(40 * n / len(ok)))}")
    print(f"\nKonsistensi prediksi: {'SAMA di semua replika' if len(outputs) == 1 else f'{len(outputs)} variasi output!'}")
    if len(outputs) == 1 and ok:
        print(f"  predictions = {ok[0][3]}")


if __name__ == "__main__":
    main()