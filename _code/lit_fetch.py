#!/usr/bin/env python3
"""文献检索/下载 CLI(Elsevier Scopus + ScienceDirect)。

🔴 凭证纪律: API key 由本脚本内部调用 get-credential.sh 取得, **绝不出现在**
    命令行参数、prompt、日志或输出里。子智能体只调用本脚本, 不接触密钥。

用法:
  python lit_fetch.py search  --query '<Scopus query>' [--count 25] [--out FILE]
  python lit_fetch.py fulltext --doi 10.xxxx/yyy --out DIR
  python lit_fetch.py abstract --doi 10.xxxx/yyy
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import urllib.error
import urllib.parse
import urllib.request

GET_CRED = Path.home() / ".claude/skills/share-docs/scripts/get-credential.sh"
SCOPUS = "https://api.elsevier.com/content/search/scopus"
ARTICLE = "https://api.elsevier.com/content/article/doi/"
TIMEOUT = 60


def _cred(name: str) -> str:
    out = subprocess.run([str(GET_CRED), name], capture_output=True, text=True, timeout=30)
    v = out.stdout.strip()
    if not v:
        raise SystemExit(f"取不到凭证 {name}（不打印任何密钥内容）")
    return v


def _headers(accept: str = "application/json") -> dict:
    return {
        "X-ELS-APIKey": _cred("ELSEVIER_API_KEY"),
        "X-ELS-Insttoken": _cred("ELSEVIER_INSTTOKEN"),
        "Accept": accept,
        "User-Agent": "volve-causal-research/0.1",
    }


def _get(url: str, accept: str = "application/json", retries: int = 3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=_headers(accept))
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (401, 403, 404):
                break            # 无权限/不存在, 重试无意义
            time.sleep(2 * (i + 1))
        except Exception as e:      # noqa: BLE001
            last = str(e)[:120]
            time.sleep(2 * (i + 1))
    raise RuntimeError(last or "unknown error")


def search(query: str, count: int = 25, start: int = 0) -> list[dict]:
    """Scopus 检索。返回精简后的条目列表(不含全文)。"""
    got, cursor = [], start
    while len(got) < count:
        n = min(25, count - len(got))
        url = (f"{SCOPUS}?query={urllib.parse.quote(query)}"
               f"&count={n}&start={cursor}&view=STANDARD&sort=-citedby-count")
        data = json.loads(_get(url))
        entries = data.get("search-results", {}).get("entry", [])
        if not entries or "error" in entries[0]:
            break
        for e in entries:
            got.append({
                "title": e.get("dc:title"),
                "doi": e.get("prism:doi"),
                "journal": e.get("prism:publicationName"),
                "year": (e.get("prism:coverDate") or "")[:4],
                "cited_by": int(e.get("citedby-count") or 0),
                "type": e.get("subtypeDescription"),
                "openaccess": e.get("openaccess") == "1",
                "scopus_id": (e.get("dc:identifier") or "").replace("SCOPUS_ID:", ""),
            })
        cursor += n
        if len(entries) < n:
            break
    return got


def _strip_xml(raw: bytes) -> str:
    t = raw.decode("utf-8", "ignore")
    t = re.sub(r"<\?xml.*?\?>", "", t, flags=re.S)
    t = re.sub(r"<(ce:)?(display|formula|inline-formula)[^>]*>.*?</\1\2>", " [FORMULA] ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&[a-z]+;", " ", t)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", t)).strip()


def fulltext(doi: str, out_dir: Path) -> dict:
    """取 ScienceDirect 全文。非 Elsevier 刊或无权限时回落到摘要。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", doi)[:100]
    for accept, tag in (("text/xml", "fulltext"), ("application/json", "abstract")):
        try:
            raw = _get(ARTICLE + urllib.parse.quote(doi, safe=""), accept)
        except Exception as e:   # noqa: BLE001
            err = str(e)
            continue
        text = _strip_xml(raw) if tag == "fulltext" else json.dumps(
            json.loads(raw), ensure_ascii=False, indent=1)
        if len(text) < 1500 and tag == "fulltext":
            continue             # 太短多半只是元数据, 试下一种
        p = out_dir / f"{slug}.{tag}.txt"
        p.write_text(text, encoding="utf-8")
        return {"doi": doi, "status": tag, "path": str(p), "chars": len(text)}
    return {"doi": doi, "status": "failed", "error": err[:120], "path": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("--query", required=True)
    s.add_argument("--count", type=int, default=25); s.add_argument("--out")
    f = sub.add_parser("fulltext"); f.add_argument("--doi", required=True)
    f.add_argument("--out", default="_refs/fulltext")
    a = sub.add_parser("abstract"); a.add_argument("--doi", required=True)
    args = ap.parse_args()

    if args.cmd == "search":
        res = search(args.query, args.count)
        payload = json.dumps(res, ensure_ascii=False, indent=1)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(payload, encoding="utf-8")
            print(f"{len(res)} 条 -> {args.out}")
        else:
            print(payload)
    elif args.cmd == "fulltext":
        print(json.dumps(fulltext(args.doi, Path(args.out)), ensure_ascii=False))
    else:
        print(_strip_xml(_get(ARTICLE + urllib.parse.quote(args.doi, safe=""),
                              "application/json"))[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
