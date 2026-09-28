"""Find which opencode zen models are usable as a teacher, and reject the ones that are not.

Guessing from model names wasted several rounds: most "-free" ids return 401 on this account, two
well-known paid ids return 500, and one produced Simplified characters inside a Traditional Chinese
answer. So every model is probed with the same short zh-TW question and scored on what comes back.

A model is only a candidate if it answers at all, answers in Chinese, and answers in *Traditional*
Chinese. Simplified leakage is disqualifying: it is a named failure mode for this project and no
amount of prompting reliably removes it.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.zen_client import ZenChat, load_key  # noqa: E402

PROBE = ("用繁體中文回答，兩句話以內：立法院委員會審查法案時，「保留，送院會處理」是什麼意思？")

# Characters that differ between Simplified and Traditional and are common in this domain.
SIMPLIFIED = set("读见后办这个来国会议员长审查处经过发对应该说话时间题问权识别绍职务专业务党团决议")
TRADITIONAL = set("讀見後辦這個來國會議員長審查處經過發對應該說話時間題問權識別紹職務專業務黨團決議")


def script_mix(text: str) -> tuple:
    simp = sum(1 for c in text if c in SIMPLIFIED and c not in TRADITIONAL)
    trad = sum(1 for c in text if c in TRADITIONAL and c not in SIMPLIFIED)
    return simp, trad


def list_models() -> list:
    req = urllib.request.Request(
        "https://opencode.ai/zen/v1/models",
        headers={"Authorization": f"Bearer {load_key()}",
                 "User-Agent": "meeting-summarizer/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    items = data if isinstance(data, list) else data.get("data", [])
    return [m["id"] for m in items if isinstance(m, dict) and m.get("id")]


def pricing() -> dict:
    """models.dev needs a browser-ish User-Agent and is optional: probing must work without it."""
    cache = "reports/models_dev.json"
    try:
        req = urllib.request.Request("https://models.dev/api.json",
                                     headers={"User-Agent": "meeting-summarizer/1.0",
                                              "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        os.makedirs("reports", exist_ok=True)
        json.dump(data, open(cache, "w"), ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        if not os.path.exists(cache):
            print(f"  pricing unavailable ({type(e).__name__}); continuing without it")
            return {}
        data = json.load(open(cache, encoding="utf-8"))
    out = {}
    for mid, m in (data.get("opencode") or {}).get("models", {}).items():
        cost, limit = m.get("cost") or {}, m.get("limit") or {}
        out[mid] = {"in": cost.get("input"), "out": cost.get("output"),
                    "ctx": limit.get("context"), "max_out": limit.get("output"),
                    "reasoning": bool(m.get("reasoning"))}
    return out


def probe(model: str, budget: int) -> dict:
    chat = ZenChat(model, max_tokens=budget, timeout=240, retries=1)
    t0 = time.time()
    reply = (chat([{"role": "user", "content": PROBE}]) or "").strip()
    elapsed = time.time() - t0
    if not reply:
        return {"model": model, "status": "fail", "detail": (chat.last_error or "empty")[:90],
                "seconds": round(elapsed, 1)}
    han = len(re.findall(r"[一-鿿]", reply))
    simp, trad = script_mix(reply)
    status = "ok"
    if han < 8:
        status = "not-chinese"
    elif simp > trad:
        status = "simplified"
    return {"model": model, "status": status, "seconds": round(elapsed, 1), "han": han,
            "simplified_chars": simp, "traditional_chars": trad, "detail": reply[:100]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/zen_probe.json")
    ap.add_argument("--budget", type=int, default=4000, help="reasoning models need room to think")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--models-from", default=None,
                    help="JSON list of {model: ...} to probe instead of every model on the gateway")
    args = ap.parse_args()

    prices = pricing()
    if args.models_from:
        models = [r["model"] for r in json.load(open(args.models_from, encoding="utf-8"))]
    else:
        models = list_models()
    print(f"probing {len(models)} models")
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(lambda m: probe(m, args.budget), models))
    for r in results:
        r.update({k: v for k, v in (prices.get(r["model"]) or {}).items()})

    ok = [r for r in results if r["status"] == "ok"]
    ok.sort(key=lambda r: (r.get("in") if r.get("in") is not None else 99, r["seconds"]))
    print(f"\nusable: {len(ok)} of {len(results)}")
    print(f"{'model':<30}{'in$/M':>7}{'out$/M':>8}{'ctx':>9}{'secs':>7}  reasoning")
    for r in ok:
        print(f"{r['model']:<30}{str(r.get('in')):>7}{str(r.get('out')):>8}"
              f"{str(r.get('ctx')):>9}{r['seconds']:>7}  {r.get('reasoning')}")
    for label in ("simplified", "not-chinese"):
        bad = [r["model"] for r in results if r["status"] == label]
        if bad:
            print(f"\nrejected ({label}): {', '.join(bad)}")
    failed = [r for r in results if r["status"] == "fail"]
    print(f"\nunavailable: {len(failed)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
