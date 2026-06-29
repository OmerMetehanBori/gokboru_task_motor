"""
GitHub API ile proje dosyalarını repository'e yükler.
Git kurulumu gerekmez — sadece Python requests kullanır.

Kullanım:
  python push_to_github.py --token <GITHUB_PAT>

Token alma: https://github.com/settings/tokens/new
  -> 'repo' iznini işaretle
"""

from __future__ import annotations

import argparse
import base64
import logging
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests kütüphanesi gerekli: pip install requests")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("GitHubPush")

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_OWNER   = "OmerMetehanBori"
REPO_NAME    = "gokboru_task_motor"
BRANCH       = "main"
API_BASE     = "https://api.github.com"

# Yüklenmeyecek dosya/klasörler
EXCLUDE_PATTERNS = {
    "__pycache__", ".git", "_visdrone_tmp",
    "*.pyc", "*.pyo", "*.tmp",
    "runs",          # eğitim çıktıları (büyük)
    "dataset",       # dataset görselleri (büyük — binlerce görüntü)
    "yolov8n.pt",    # eski model dosyası
}


def should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_PATTERNS:
            return True
    name = path.name
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*") and name.endswith(pattern[1:]):
            return True
        if name == pattern:
            return True
    return False


def collect_files() -> list[Path]:
    files = []
    for f in PROJECT_ROOT.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(PROJECT_ROOT)
        if should_exclude(rel):
            continue
        files.append(f)
    return sorted(files)


def get_file_sha(session: requests.Session, repo_path: str) -> str | None:
    """Dosya zaten varsa SHA'sını döner (update için gerekli)."""
    url = f"{API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    r = session.get(url)
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def upload_file(session: requests.Session, local_path: Path, repo_path: str) -> bool:
    url = f"{API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    try:
        content = local_path.read_bytes()
    except Exception as e:
        log.warning("Okunamadı: %s — %s", local_path.name, e)
        return False

    content_b64 = base64.b64encode(content).decode("utf-8")
    sha = get_file_sha(session, repo_path)

    payload: dict = {
        "message": f"add {repo_path}",
        "content": content_b64,
        "branch": BRANCH,
    }
    if sha:
        payload["message"] = f"update {repo_path}"
        payload["sha"] = sha

    r = session.put(url, json=payload)
    if r.status_code in (200, 201):
        action = "güncellendi" if sha else "yüklendi"
        log.info("  ✓ %s %s", repo_path, action)
        return True
    else:
        log.error("  ✗ %s — HTTP %d: %s", repo_path, r.status_code, r.text[:120])
        return False


def ensure_gitignore(session: requests.Session) -> None:
    """Yoksa .gitignore oluşturur."""
    gitignore = PROJECT_ROOT / ".gitignore"
    if not gitignore.exists():
        content = """\
__pycache__/
*.pyc
*.pyo
*.pt
runs/
dataset/
outputs/
models/
_visdrone_tmp/
*.zip
*.tmp
"""
        gitignore.write_text(content, encoding="utf-8")
    upload_file(session, gitignore, ".gitignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="GitHub'a proje dosyalarını yükle")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument("--dry-run", action="store_true", help="Sadece listeyi göster, yükleme yapma")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({
        "Authorization": f"token {args.token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })

    # Token kontrolü
    me = session.get(f"{API_BASE}/user")
    if me.status_code != 200:
        log.error("Token geçersiz veya yetkisiz: HTTP %d", me.status_code)
        sys.exit(1)
    log.info("Giriş yapıldı: %s", me.json().get("login"))

    # Repo kontrolü
    repo_check = session.get(f"{API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}")
    if repo_check.status_code != 200:
        log.error("Repository bulunamadı: %s/%s", REPO_OWNER, REPO_NAME)
        sys.exit(1)
    log.info("Repository: %s/%s", REPO_OWNER, REPO_NAME)

    files = collect_files()
    log.info("\nYüklenecek %d dosya:", len(files))
    for f in files:
        rel = f.relative_to(PROJECT_ROOT)
        size_kb = f.stat().st_size / 1024
        log.info("  %s  (%.1f KB)", rel, size_kb)

    if args.dry_run:
        log.info("\n--dry-run modu: Yükleme yapılmadı.")
        return

    log.info("\n.gitignore hazırlanıyor...")
    ensure_gitignore(session)

    log.info("\nDosyalar yükleniyor...")
    success = 0
    failed = 0
    for f in files:
        rel_path = f.relative_to(PROJECT_ROOT).as_posix()
        ok = upload_file(session, f, rel_path)
        if ok:
            success += 1
        else:
            failed += 1
        time.sleep(0.3)  # API rate limit

    log.info("\n" + "=" * 50)
    log.info("Tamamlandı: %d başarılı, %d başarısız", success, failed)
    log.info("Repository: https://github.com/%s/%s", REPO_OWNER, REPO_NAME)
    log.info("=" * 50)


if __name__ == "__main__":
    main()
