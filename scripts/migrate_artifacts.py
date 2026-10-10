import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

import requests

API = "https://api.github.com"
DEFAULT_REPO = "sulthanrps/MLOps-Youtube-Trending-Video-Prediction"
TARGETS = {
    "raw_trending_": Path("data/raw"),
    "processed_trending_": Path("data/processed"),
}


def make_session() -> requests.Session:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Token tidak ditemukan. Set env GH_TOKEN atau GITHUB_TOKEN dulu.")
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    return s


def list_artifacts(session: requests.Session, repo: str):
    page = 1
    while True:
        r = session.get(
            f"{API}/repos/{repo}/actions/artifacts",
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        r.raise_for_status()
        items = r.json().get("artifacts", [])
        if not items:
            return
        yield from items
        page += 1


def download_zip(session: requests.Session, repo: str, artifact_id: int) -> bytes:
    # requests otomatis membuang header Authorization saat di-redirect
    # ke storage GitHub (beda host), jadi aman.
    r = session.get(
        f"{API}/repos/{repo}/actions/artifacts/{artifact_id}/zip",
        timeout=300,
    )
    if r.status_code in (401, 403):
        sys.exit(
            f"Akses ditolak ({r.status_code}) saat mengunduh artifact.\n"
            "Token Codespaces kemungkinan tidak punya izin Actions. Buat fine-grained\n"
            "personal access token (akses ke repo ini, permission Actions: Read-only),\n"
            "lalu jalankan:  export GH_TOKEN=github_pat_xxx  dan ulangi skrip ini."
        )
    r.raise_for_status()
    return r.content


def target_folder(filename: str):
    if not filename.endswith(".parquet"):
        return None
    for prefix, folder in TARGETS.items():
        if filename.startswith(prefix):
            return folder
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO))
    ap.add_argument("--name-filter", default="", help="hanya artifact yang namanya mengandung teks ini")
    ap.add_argument("--dry-run", action="store_true", help="tampilkan rencana tanpa menyalin file")
    args = ap.parse_args()

    for folder in TARGETS.values():
        folder.mkdir(parents=True, exist_ok=True)

    session = make_session()
    artifacts = [
        a for a in list_artifacts(session, args.repo)
        if not a["expired"] and args.name_filter in a["name"]
    ]
    artifacts.sort(key=lambda a: a["created_at"])
    print(f"Repo: {args.repo} | artifact aktif: {len(artifacts)}"
          f"{' | DRY RUN' if args.dry_run else ''}\n")

    totals = {"copied": 0, "skipped": 0, "unknown": 0}
    for a in artifacts:
        data = download_zip(session, args.repo, a["id"])
        copied = skipped = 0
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                name = Path(member).name
                folder = target_folder(name)
                if folder is None:
                    if name.endswith(".parquet"):
                        print(f"   ? tidak dikenali, dilewati: {member}")
                        totals["unknown"] += 1
                    continue
                dest = folder / name
                if dest.exists():
                    skipped += 1
                    continue
                if not args.dry_run:
                    dest.write_bytes(zf.read(member))
                copied += 1
        totals["copied"] += copied
        totals["skipped"] += skipped
        print(f"[{a['created_at']}] {a['name']} (id {a['id']}): "
              f"{copied} disalin, {skipped} sudah ada")

    verb = "akan disalin" if args.dry_run else "disalin"
    print(f"\nSelesai: {totals['copied']} file {verb}, "
          f"{totals['skipped']} dilewati (sudah ada), "
          f"{totals['unknown']} tidak dikenali.")
    for prefix, folder in TARGETS.items():
        n = len(list(folder.glob(f"{prefix}*.parquet")))
        print(f"  {folder}/: {n} file {prefix}*.parquet")


if __name__ == "__main__":
    main()