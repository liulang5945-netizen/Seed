"""Pre-fill the runtime download cache via CN-reachable mirrors, using curl."""
import json
import os
import subprocess
import hashlib

LOCK = r"E:/Seed/taiji-harness/scripts/primary-runtime/lock.json"
DEST = r"E:/Seed/taiji-harness/apps/desktop/.desktop-build/downloads"
TARGET = "win-x64"
os.makedirs(DEST, exist_ok=True)
lock = json.load(open(LOCK, encoding="utf-8"))
t = lock["targets"][TARGET]
CURL = r"C:\Windows\System32\curl.exe"

def fetch(urls, sha, name):
    dest = os.path.join(DEST, sha)
    if os.path.exists(dest):
        print(f"skip (cached)  {name}")
        return
    for url in urls:
        print(f"try  {url}")
        r = subprocess.run([CURL, "-L", "--fail", "--connect-timeout", "20",
                            "--max-time", "600", "-s", "-o", dest + ".part", url])
        if r.returncode != 0:
            print(f"  curl rc={r.returncode} — next mirror")
            continue
        h = hashlib.sha256(open(dest + ".part", "rb").read()).hexdigest()
        if h != sha:
            print(f"  checksum mismatch {h[:12]} — next mirror")
            continue
        os.replace(dest + ".part", dest)
        print(f"OK  {name}  {os.path.getsize(dest)/1e6:.1f} MB")
        return
    raise SystemExit(f"ALL MIRRORS FAILED for {name}")

nv = lock["nodeVersion"]
node_file = f"node-v{nv}-{t['nodeArchive']}"
fetch([
    f"https://cdn.npmmirror.com/binaries/node/v{nv}/{node_file}",
    f"https://nodejs.org/dist/v{nv}/{node_file}",
], t["nodeSha256"], f"node v{nv} {t['nodeArchive']}")

py_name = f"cpython-{lock['pythonVersion']}+{lock['pythonRelease']}-{t['pythonTarget']}-install_only_stripped.tar.gz"
fetch([
    f"https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download/{lock['pythonRelease']}/{py_name}",
    f"https://github.com/astral-sh/python-build-standalone/releases/download/{lock['pythonRelease']}/{py_name}",
], t["pythonSha256"], py_name)

for w in list(t["wheels"]) + list(lock["wheels"]):
    fn = w["url"].rsplit("/", 1)[-1]
    tuna = w["url"].replace("https://files.pythonhosted.org/", "https://pypi.tuna.tsinghua.edu.cn/")
    fetch([tuna, w["url"]], w["sha256"], fn)

print("cache prefill complete ->", DEST)
