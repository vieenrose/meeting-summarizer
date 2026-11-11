"""2026-11-11 questionnaire metric checks: (1) repeat noise = |score - rescoring| of the same summaries; (2) per-session spread of the score across
seeds of one model (what best-of-K could gain); (3) session-level correlation with the old coverage; (4) paired control vs rdsft3p1."""
import json, os, statistics as S
def load(c):
    f = f"reports/qa/{c}.json"
    return {r["id"]: r for r in json.load(open(f))["rows"] if "v" in r and not r.get("missing")} if os.path.exists(f) else {}
sc = lambda r: (r["v"].count(1) - r["v"].count(-1)) / r["n"]; co = lambda r: r["v"].count(1) / r["n"]
for T in ["msdev", "q12", "ali20"]:
    a, b = load(f"rdsft1p1u85s2-{T}"), load(f"rdsft1p1u85s2-{T}.rep")
    rep = [abs(co(a[i]) - co(b[i])) for i in a if i in b]; flip = [sum(x != y for x, y in zip(a[i]["v"], b[i]["v"])) / a[i]["n"] for i in a if i in b and a[i]["n"] == b[i]["n"]]
    for M in ["rdsft1p1u85", "rdsft3p1"]:
        runs = [load(f"{M}s{k}-{T}") for k in (2, 3, 4, 5)]; ids = set.intersection(*[set(r) for r in runs]) if all(runs) else set()
        if not ids: continue
        sd = [S.pstdev([co(r[i]) for r in runs]) for i in ids]; rng = [max(co(r[i]) for r in runs) - min(co(r[i]) for r in runs) for i in ids]
        best = S.mean(max(co(r[i]) for r in runs) for i in ids); mean = S.mean(S.mean(co(r[i]) for r in runs) for i in ids); nq = S.mean(runs[0][i]["n"] for i in ids)
        print(f"{T} {M}: sessions {len(ids)} q/session {nq:.0f} | mean correct {mean:.3f} best-of-4 {best:.3f} | within-session sd {S.mean(sd):.3f} range {S.mean(rng):.3f} | binomial sd expected {S.mean((co(runs[0][i]) * (1 - co(runs[0][i])) / runs[0][i]['n']) ** .5 for i in ids):.3f}")
    print(f"{T} repeat noise: mean |diff| {S.mean(rep) if rep else float('nan'):.3f} verdict flips {S.mean(flip) if flip else float('nan'):.3f}")
    old = {r["id"]: r["coverage"] for r in json.load(open(f"reports/coverage/rdsft1p1u85s2-{T}.json"))["rows"]}; x = [(co(a[i]), old[i]) for i in a if i in old]
    if len(x) > 2: print(f"{T} session correlation qa vs old coverage: {S.correlation([p for p, _ in x], [q for _, q in x]):.2f}")
