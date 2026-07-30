"""Exact Python port of VoxSumDroid SummaryText.chunk / groupPartials / foldToFit
and the Summarizer/ActionItemExtractor char budgets — the student must be trained on
the same windows the app will feed it.
"""
N_CTX = 4096


def map_budget(out_tokens: int) -> int:
    """Summarizer: ((nCtx - outTokens - 96) * 3 / 5).coerceIn(512, 3500)"""
    return max(512, min(3500, (N_CTX - out_tokens - 96) * 3 // 5))


def chunk(text: str, size: int = 3500, overlap: int = 300) -> list:
    sz = max(size, 1)
    ov = min(max(overlap, 0), sz - 1)
    if len(text) <= sz:
        return [text]
    out = []
    start = 0
    while start < len(text):
        end = min(start + sz, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - ov
    return out


def group_partials(partials: list, budget_chars: int) -> list:
    groups, cur, cur_len = [], [], 0
    for p in partials:
        sep = 0 if not cur else 2
        if cur and cur_len + sep + len(p) > budget_chars:
            groups.append(cur)
            cur, cur_len = [], 0
        cur_len += (0 if not cur else 2) + len(p)
        cur.append(p)
    if cur:
        groups.append(cur)
    return groups


def fold_to_fit(partials: list, budget_chars: int, separator: str, reduce_group) -> list:
    level = partials
    while len(level) > 1 and len(separator.join(level)) > budget_chars:
        folded = False
        nxt = []
        for group in group_partials(level, budget_chars):
            if len(group) == 1:
                nxt.append(group[0])
                continue
            folded = True
            nxt.append(reduce_group(group))
        level = nxt
        if not folded:
            break
    return level
