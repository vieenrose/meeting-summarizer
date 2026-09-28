"""Build a held-out evaluation split for corpus v2, same spirit as v1's data/split.json: spread
across committees and session types so held-out performance isn't an artifact of one domain,
and reproducible (seeded) rather than hand-picked.

v1 held out 7/39 (18%) sessions, "one per committee where possible, mixing budget review,
clause-by-clause markup and general meetings." This follows the same rule at v2's scale.
"""
import argparse
import collections
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/ivod_v2_manifest.json")
    ap.add_argument("--rows", default="data/train/v2_rows.jsonl",
                    help="only sessions that actually made it into the exported rows are eligible")
    ap.add_argument("--fraction", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/split_v2.json")
    args = ap.parse_args()

    exported = {json.loads(l)["session"] for l in open(args.rows, encoding="utf-8")}
    manifest = {f"ivod_{s['ivod_id']}": s
                for s in json.load(open(args.manifest, encoding="utf-8"))["sessions"]}

    by_group = collections.defaultdict(list)
    for sid in sorted(exported):
        m = manifest.get(sid, {})
        by_group[(m.get("committee", "?"), m.get("type", "?"))].append(sid)

    rng = random.Random(args.seed)
    target_total = round(len(exported) * args.fraction)
    heldout = []
    groups = list(by_group.items())
    rng.shuffle(groups)
    # Round-robin across (committee, type) groups so every combination gets representation before
    # any single one is over-sampled, same "one per committee where possible" spirit as v1.
    group_pools = {k: rng.sample(v, len(v)) for k, v in groups}
    while len(heldout) < target_total and any(group_pools.values()):
        for k, _ in groups:
            if len(heldout) >= target_total:
                break
            if group_pools[k]:
                heldout.append(group_pools[k].pop())

    heldout.sort()
    by_committee = collections.Counter(manifest.get(sid, {}).get("committee", "?") for sid in heldout)
    out = {
        "note": ("Held-out sessions for student evaluation, corpus v2. Stratified by "
                 "(committee, session type), seeded and reproducible (scripts/build_split.py "
                 f"--seed {args.seed}). {len(heldout)}/{len(exported)} sessions "
                 f"({len(heldout) / len(exported):.0%}). Never train on these."),
        "heldout": heldout,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(heldout)}/{len(exported)} held out -> {args.out}")
    print("by committee:", dict(by_committee))


if __name__ == "__main__":
    main()
