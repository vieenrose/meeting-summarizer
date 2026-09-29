"""Restore citations the verification pass dropped, on runs made before the agent kept them:
each verified item without a citation takes the citations of the most similar draft item."""
import json
import os
import re
import sys

CITE = re.compile(r"\[\d+:\d{2}(?::\d{2})?\]")


def bigrams(t):
    t = re.sub(r"\s|\[[^\]]*\]", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def main(run):
    sys.path.insert(0, ".")
    from eval.journal_agent import as_prose
    fixed = total = 0
    for f in sorted(os.listdir(run)):
        if not f.endswith(".json"):
            continue
        r = json.load(open(os.path.join(run, f), encoding="utf-8"))
        draft = [l for l in json.load(open(os.path.join(run + "-draft", f), encoding="utf-8"))["minutes"].splitlines()
                 if l.startswith("- ") and CITE.search(l)]
        out = []
        for line in r["minutes"].splitlines():
            if line.startswith("- ") and line[2:].strip() != "無" and not CITE.search(line):
                total += 1
                b = bigrams(line)
                best = max(draft, key=lambda d: len(b & bigrams(d)) / max(1, len(b | bigrams(d))), default=None)
                if best:
                    line = line.rstrip("。 ") + " " + " ".join(CITE.findall(best))
                    fixed += 1
            out.append(line)
        r["minutes"] = "\n".join(out)
        r["prose"] = as_prose(r["minutes"])
        json.dump(r, open(os.path.join(run, f), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{run}: restored citations on {fixed}/{total} uncited items")


if __name__ == "__main__":
    main(sys.argv[1])
