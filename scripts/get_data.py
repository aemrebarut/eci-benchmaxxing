"""Download Epoch AI's benchmark data zip and record which snapshot we used.

Epoch republishes benchmark_data.zip as new results arrive, so the URL does
not serve old versions. data/SNAPSHOT.json records the sha256 of the copy
our analysis ran on; --check verifies a local copy against it.
"""
import argparse, datetime, hashlib, json, pathlib, urllib.request

URL = "https://epoch.ai/data/benchmark_data.zip"
ROOT = pathlib.Path(__file__).resolve().parent.parent
ZIP = ROOT / "data" / "benchmark_data.zip"
MANIFEST = ROOT / "data" / "SNAPSHOT.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify data/benchmark_data.zip against data/SNAPSHOT.json")
    ap.add_argument("--record", action="store_true", help="write data/SNAPSHOT.json for the local zip without downloading")
    args = ap.parse_args()

    if args.check:
        want = json.loads(MANIFEST.read_text())["sha256"]
        got = sha256(ZIP)
        print("ok" if got == want else f"MISMATCH: local {got}, snapshot {want}")
        return

    if not args.record:
        ZIP.parent.mkdir(exist_ok=True)
        tmp = ZIP.with_suffix(".zip.part")
        with urllib.request.urlopen(URL) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.replace(ZIP)

    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps({
        "url": URL,
        "retrieved_at": now if not args.record else datetime.datetime.fromtimestamp(ZIP.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
        "bytes": ZIP.stat().st_size,
        "sha256": sha256(ZIP),
        "license": "CC-BY 4.0, Epoch AI",
    }, indent=2) + "\n")
    print(MANIFEST.read_text())


if __name__ == "__main__":
    main()
