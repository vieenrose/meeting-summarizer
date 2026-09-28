"""Repair the semantic defects a mechanical checker cannot see.

The structural validator only knows whether a citation exists, whether the point is short
enough, and whether all three thirds of the meeting are represented. It cannot tell that a
point is fluent, correctly formatted, well cited and still wrong. A semantic review of all
39 gold summaries against their transcripts found nine such points; this repairs them.

Every replacement below is grounded in a transcript line quoted in `evidence`, and the
corrected wording is taken from what the speakers actually said, not composed freely. Where
the transcript is too garbled to establish the fact, the claim is weakened to what the text
supports rather than guessed at -- an unresolvable line belongs in `limits`, not in a
confident sentence.

Each change is appended to the session's `repairs` list, so the review site shows what was
altered and why alongside the data itself.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "..", "runs", "gold", "w4000")

# Round 1: findings against the numbered points, before the prose note existed.
# session -> list of (point number (1-based), kind, old fragment, new fragment, why)
# An `old` of None replaces the whole point body.
ROUND1 = {
    "ivod_16479": [(
        2, "CITATION DRIFT",
        "[39:42][36:22]", "[38:22][38:43]",
        "35.47% 與住宿會館整修的說明在 38:22-38:43；39:42 是報告開頭，36:22 是另一筆 95.38%。",
    )],
    "ivod_17242": [(
        4, "CITATION DRIFT",
        "1:07:46僑委會副委員長表示該專案自去年9月底至1月23日共6件申請，皆來自越南臺商，累計授信166萬美元、保證149萬美元。[44:23][1:07:46]",
        "46:35起說明該專案自去年9月底至1月23日共6件申請，累計保證金額149萬美元；1:07:46補充6件皆為越南臺商。[44:23][46:56][1:07:46]",
        "166萬/149萬兩項金額在 46:35-46:56，不在 1:07:46；1:07:46 只說了六件皆來自越南臺商。",
    )],
    "ivod_17409": [(
        4, "NUMBER ERROR",
        "第27案辦理農民運運普查減列2億9476萬1千元",
        "第27案辦理農民普查改凍結100萬元並提書面報告",
        "2億9476萬1千元是 42:11 的原提案數；1:51:48 通過的決議是改凍結100萬元。",
    )],
    "ivod_17421": [
        (
            2, "CITATION DRIFT", "[25:34]", "[25:34][29:39]",
            "黃建威於 29:14 才被請發言，其反對議程的發言在 29:39 起；25:34 是另一位發言者。",
        ),
        (
            3, "CITATION DRIFT", "[1:08:51]", "[1:20:01][1:08:51]",
            "411,731人與七成集中六都在 1:20:01-1:20:19；1:08:51 是黃建榮談34個團體。",
        ),
    ],
    "ivod_17452": [(
        5, "FABRICATION",
        "最終決議減列150萬元（區域政委員50萬加歐委員100萬）、凍結100萬元，併案處理。 [1:28:00]",
        "最終決議取消減列、凍結5%（約7萬1千元），解凍條件為提出書面報告。 [1:09:27][1:10:27]",
        "減列150萬、凍結100萬是 1:28:00 另一案（40人出國訓練）的決議；第15案在 1:09:27-1:10:27 的決議是取消簡列、凍結5%約7萬1千元。",
    )],
    "ivod_17478": [(
        4, "MISATTRIBUTION",
        "308、317、325改為主決議，凍結5000萬元；",
        "308、317、325改為主決議；",
        "1:07:39 只說三案改為主決議；同句的凍結5000萬屬於農業管理，該筆已列於本點末。",
    )],
    "ivod_14064": [(
        6, "REVERSAL",
        "柯建銘委員批評國民黨今日未出席修憲委員會，質疑國民黨「一黨修憲」",
        "柯建銘委員指出國民黨今日未出席修憲委員會，其餘三黨均到場參與",
        "1:38:39 的逐字稿自相矛盾（「國民黨一黨來黨修憲」與「三黨都來了」並存），只有「國民黨缺席、三黨到場」是文本確實支援的部分。",
    )],
    "ivod_17541": [(
        3, "NUMBER ERROR",
        "另關切新竹至桃園高鐵電力排程造成15萬列車延誤事件。[40:46] [42:43]",
        "另關切新竹桃園段電力異常造成4班列車延誤約5分鐘。[40:46][44:46]",
        "「15萬列車」是 42:43 的辨識錯誤；44:46-45:07 說明實際是4班列車延誤5分鐘。",
    )],
}


# Round 2: the same review re-run over the repaired corpus, now covering the prose note too.
# Entries carry a surface: "summary" (default, by point number) or "prose" (point is None).
ROUND2 = {
    "ivod_16479": [(
        None, "CITATION DRIFT",
        "執行率僅35.47% [39:42][36:22]",
        "執行率僅35.47% [38:22][38:43]",
        "散文沿用了第一輪已在編號摘要修正掉的錯誤出處；35.47% 與會館整修的說明在 38:22-38:43。",
        "prose",
    )],
    "ivod_14064": [(
        None, "MISATTRIBUTION",
        "洪昇漢委員指出臺灣是亞洲唯一未落實十八歲公民權的民主國家[47:08]",
        "有委員指出臺灣在亞洲國家中是少數未落實十八歲公民權者[1:07:22]",
        "47:08 是另一位委員說明第23案與第50案，未提此事；「亞洲唯一」的說法出自 1:07:22 的另一位發言者，逐字稿的發言者標籤不可靠，故不指名。",
        "prose",
    )],
    "ivod_17380": [(
        6, "CITATION DRIFT", "[1:18:01]", "[2:13:33]",
        "1:18:01 是「暫時保留、撤回」的臨時處置；送出委員會協商的裁示在 2:13:33，教師自己的筆記也標為 DECISION。",
        "summary",
    )],
    "ivod_17449": [(
        4, "CITATION DRIFT", "[1:22:34][1:40:19]", "[1:22:34][1:26:29][1:40:19]",
        "本點含三個子句，出處卻集中在句尾兩處，第3案「改書面、凍結10%」的出處 1:26:29 未被引用。",
        "summary",
    )],
    "ivod_17652": [
        (
            4, "REVERSAL",
            "第六條、第六條之一、第七條、第九條均按行政院版本通過；第三條之一、第五條維持現行條文不修正",
            "第六條、第六條之一、第九條均按行政院版本通過；第三條之一、第五條、第七條維持現行條文不修正",
            "1:00:16-1:00:22 就第七條「建議不要修正」「維持原規」，未依行政院版通過。",
            "summary",
        ),
        (
            None, "REVERSAL",
            "第六條、第六條之一、第七條、第九條均依行政院版本通過[55:29][56:36][57:55][1:01:59]；第五條、第三條之一維持現行條文不予修正[55:14][54:04]",
            "第六條、第六條之一、第九條均依行政院版本通過[55:29][56:36][1:01:59]；第五條、第三條之一、第七條維持現行條文不予修正[55:14][54:04][1:00:21]",
            "散文段落重複了同一個錯誤：第七條在 1:00:21 維持原規定，不在通過之列。",
            "prose",
        ),
    ],
    "ivod_17698": [(
        3, "NUMBER ERROR",
        "第五、十一、十三、十五、十六條不予修正",
        "第五、十一、十三、十、十六條不予修正",
        "第十五條已在 32:36 修正通過，不可能同時列為不予修正；34:19 的不修正清單為第十及第十六條。",
        "summary",
    )],
}

ROUNDS = {1: ROUND1, 2: ROUND2}


# Notes are the source every summary is rendered from, and they are also the per-window
# training rows. A defect left here survives regeneration and ships in the training data,
# so it is repaired at this level too. Keyed by note timestamp rather than index, because
# indices shift. Surface "note".
NOTES = {
    "ivod_17652": [(
        "57:55", "REVERSAL",
        "第七條部分，按行政院版本通過（加速折舊條文已移至初審條例處理",
        "第六條之一按行政院版本通過，接著進入第七條討論（加速折舊條文已移至初審條例處理",
        "57:55 通過的是第六條之一，第七條才剛開始討論；其結果在 1:00:21 為維持原規，同場 1:33:57 的協商結論也記為「第七條不徵定」。",
        "note",
    )],
    "ivod_17698": [(
        "32:36", "NUMBER ERROR",
        "第五條、第十一條、第十三條、第十五條、第十六條均不予修正",
        "第五條、第十一條、第十三條、第十條、第十六條均不予修正",
        "第十五條已在同一則筆記中記為修正通過，不可能同時不予修正；34:19 的清單為第十及第十六條。",
        "note",
    )],
    "ivod_17478": [(
        "1:07:39", "MISATTRIBUTION",
        "308、317、325均改為主決議，凍結5000萬元，要求",
        "308、317、325均改為主決議；農業管理凍結5000萬元，要求",
        "逐字稿在同一句裡先宣布三案改為主決議，再轉到農業管理才講凍結5000萬，兩者不屬同一案。",
        "note",
    )],
    "ivod_17541": [(
        "42:43", "NUMBER ERROR",
        "因電力排程造成15萬列車延誤5至10分鐘之事",
        "因電力異常造成4班列車延誤約5分鐘之事",
        "「15萬」是 42:43 的語音辨識錯誤，44:46-45:07 說明實際為4班列車延誤5分鐘。",
        "note",
    )],
}

ROUNDS[3] = NOTES


# Round 4: findings from the third review, which was the first to read the notes themselves.
# Surface "note-ts" corrects a note's own timestamp: a note IS its citation, so a note filed
# under the wrong minute mis-cites every summary written from it.
ROUND4 = {
    "ivod_14064": [(
        "1:48:46", "MISATTRIBUTION",
        "柯建銘總召（國民黨團）: 發表修憲立場演說，批評民進黨退席是「暴力表決」，指責對方「一派胡言」稱修憲是惡習",
        "柯建銘總召（民主進步黨黨團）: 發表修憲立場演說，回應外界對該黨退席的批評，駁斥「暴力表決」與修憲是惡習的說法為「一派胡言」",
        "1:35:18 該發言者自述「代表民主進步黨黨團」，1:50:41 主席稱其為柯總召；他是在反駁對自己黨團退席的指控，不是批評民進黨。",
        "note",
    )],
    "ivod_17211": [
        (
            "28:33", "MISATTRIBUTION",
            "國防部勤次（朱惠民）因參加專業會議請假",
            "國防部勤次因參加專業會議請假",
            "28:33 只以職稱「勤次」稱呼請假者，未具名；朱惠民少將是代理出席者，不是請假者。",
            "note",
        ),
        (
            4, "MISATTRIBUTION",
            "國防部勤次朱惠民因參加專業會議請假，准假；今日由助理次長朱惠民少將代理出席",
            "國防部勤次因參加專業會議請假，准假；今日由助理次長朱惠民少將代理出席",
            "同一人不會既請假又代理自己出席；逐字稿未具名請假者。",
            "summary",
        ),
        (
            None, "MISATTRIBUTION",
            "國防部勤次朱惠民因參加專業會議請",
            "國防部勤次因參加專業會議請",
            "散文沿用同一錯誤。",
            "prose",
        ),
    ],
    "ivod_16376": [(
        4, "CITATION DRIFT",
        "[1:29:48][1:33:37]", "[1:29:48][1:31:16][1:32:41]",
        "1:33:37 是另一案 9:8:1 的表決；民進黨提案兩次 8:9:1 的表決在 1:31:16 與 1:32:41。",
        "summary",
    )],
    "ivod_16479": [(
        "39:42", "CITATION DRIFT", "39:42", "34:28",
        "該筆記的內容（歲入1,380.2萬元、執行率88.26%）在 34:28-35:47 宣讀；39:42 是後面談報告備查的另一句。",
        "note-ts",
    )],
    "ivod_17478": [(
        None, "MISATTRIBUTION",
        "第3案農業管理部分308、317、325均改為主決議，凍結5000萬元[1:07:39]",
        "第3案農業管理部分308、317、325均改為主決議[1:07:39]",
        "散文重新合併了筆記與編號摘要已修正的兩件事；凍結5000萬屬農業管理，已列於本段末的協商結論。",
        "prose",
    )],
    "ivod_17490": [(
        None, "NUMBER ERROR",
        "第40至45案以第45案為主案減列650萬元",
        "第41至45案以第45案為主案減列650萬元",
        "1:10:53 明言「這是41，42到45」；編號摘要第4點已寫第41至45案，散文與之矛盾。",
        "prose",
    )],
    "ivod_17449": [
        (
            2, "MISATTRIBUTION",
            "第45、48、53案凍結300萬，3個月內書面報告後動支",
            "第48案凍結300萬，3個月內書面報告後動支",
            "56:30 顯示 300萬 與文字修正屬第48案；第53案為凍結14%，第45案改書面，本點稍後也自相矛盾地分別記載了這兩項。",
            "summary",
        ),
        (
            None, "MISATTRIBUTION",
            "第45、48、53案凍結300萬，3個月內書面報告[57:20]",
            "第48案凍結300萬，3個月內書面報告[57:20]",
            "散文沿用同一錯誤分組。",
            "prose",
        ),
    ],
    "ivod_17541": [
        (
            "1:00:37", "MISATTRIBUTION",
            "第23案（鐵道局工程影響行車安全案）決議凍結預算1000萬",
            "鐵道局工程影響行車安全案決議凍結預算1000萬",
            "1:00:37 未說出案號；第23案是 1:00:54 另一位委員提出的高雄捷運延伸案。",
            "note",
        ),
        (
            2, "MISATTRIBUTION",
            "第23案凍結1000萬", "鐵道局行車安全案凍結1000萬",
            "同上：該筆凍結在逐字稿中沒有案號。",
            "summary",
        ),
    ],
}

ROUNDS[4] = ROUND4


# Round 5: the low-confidence flags from round 3, adjudicated one at a time.
ROUND5 = {
    "ivod_13781": [(
        "54:26", "NUMBER ERROR",
        "預計分五年辦理，今年編列第1年經費110億元",
        "預計分五年辦理，今年編列第1年經費",
        "55:08 的「110億元」大於立法院全年總經費（約36.7億），不可能為真，應是「110年度」的辨識錯誤；"
        "與其複述一個不可能的數字，不如只記可查證的部分。第二筆低信心標記（1:21:24 的1.3億）未修改，"
        "因為那是委員本人在質詢中提出的說法，筆記如實轉述並無錯誤。",
        "note",
    )],
}

ROUNDS[5] = ROUND5


# Round 6: round 4's findings, plus a convention this project had been applying two ways.
#
# A note summarises a speaker's whole turn, so it is filed at the turn's start even when a
# figure inside it is spoken a minute later. A summary point or a prose sentence makes ONE
# claim, so it cites the line carrying that claim. Earlier rounds repaired some summaries to
# the precise line while leaving the parallel prose on the turn start; that inconsistency is
# resolved here in favour of precision for the derived surfaces, and turn-start for notes.
ROUND6 = {
    "ivod_14064": [(
        "1:38:39", "REVERSAL",
        "批評國民黨今日未出席，質疑國民黨「一黨修憲」，三黨中僅國民黨缺席",
        "指出國民黨今日未出席，其餘三黨均到場參與修憲",
        "第一輪已據此修正摘要與散文，但來源筆記未改，仍留著逐字稿自相矛盾處（「國民黨一黨來黨修憲」與「三黨都來了」）中不可靠的那一半。",
        "note",
    )],
    "ivod_17242": [(
        None, "CITATION DRIFT",
        "累計授信166萬美元、保證149萬美元[45:27][1:17:27]",
        "累計授信166萬美元、保證149萬美元[46:56][1:17:27]",
        "金額在 46:56 說出。筆記依發言輪起點編號無妨，但摘要與散文的單一主張應引用承載該主張的那一行；"
        "本案的編號摘要已於第一輪如此修正，散文一併對齊。",
        "prose",
    )],
}

ROUNDS[6] = ROUND6


# Round 7: round 4's second batch. Four repairs that had reached one surface but not the
# other, plus one the reviewer reported inverted -- see ivod_17595.
ROUND7 = {
    "ivod_17409": [(
        None, "NUMBER ERROR",
        "第27案農民運運普查減列2億9476萬1千元[42:32]",
        "第27案農民普查改凍結100萬元並提書面報告[1:51:48]",
        "2億9476萬1千元是 42:11 的原提案數，通過的決議是 1:51:48 的改凍結100萬元；"
        "編號摘要第一輪已改，散文未同步。參考逐字稿同段記為「第二十七案 辦理...改凍結一百萬元」。",
        "prose",
    )],
    "ivod_17410": [(
        None, "CITATION DRIFT",
        "第十二條最終修正條文於無異議下通過 [1:20:28]",
        "第十二條最終修正條文於無異議下通過 [1:43:54]",
        "1:20:28 主席只是把該條交付繕印、先處理下一條；表決通過在 1:43:54，"
        "參考逐字稿記為「我們第12條我們就依照...共同所提修正通過」。",
        "prose",
    )],
    "ivod_17421": [(
        5, "CITATION DRIFT", "[1:20:34]", "[1:25:06]",
        "薛志明參事 1:23:56 才被請發言，憲法法庭專屬權責的說法在 1:25:06-1:25:27；"
        "1:20:34 是另一位官員談行政機關不是憲法解釋機關。",
        "summary",
    )],
    "ivod_17491": [
        (
            2, "NUMBER ERROR",
            "第1至7案及第9案合併減列業務費300萬元 [42:13]",
            "第1至7案及第9案合併減列業務費360萬元 [1:06:37]",
            "300萬是協商中途的數字；1:06:37 定案為「合併刪減數360萬」，參考逐字稿同段可證。",
            "summary",
        ),
        (
            None, "NUMBER ERROR",
            "業務費減列300萬元[45:10]", "業務費減列360萬元[1:06:37]",
            "同上，散文沿用了中途數字。",
            "prose",
        ),
    ],
    "ivod_17595": [(
        "2:04:04", "REVERSAL",
        "土地銀行第3案減列1000萬元", "土地銀行第3案凍結1000萬元",
        "審查意見說第3案應為減列、摘要寫凍結，方向相反。逐字稿 2:04:04 為「第3案改動，減1000萬，"
        "3個月內提出面報告，使得東芝」——「使得東芝」即始得動支，只有凍結才附解凍條件；"
        "參考逐字稿記為「第三案改凍結一千萬三個月內提書面報告使得凍資」。錯的是筆記，不是摘要。",
        "note",
    )],
}

ROUNDS[7] = ROUND7


# Round 8: round 5's findings. The first round where reviewers held the reference themselves.
ROUND8 = {
    "ivod_17409": [
        ("24:23", "NUMBER ERROR", "1995年（應為民國114年）", "1995年（應為民國115年）",
         "逐字稿把年度讀成「1995年」，筆記補註的年份猜錯了：同場 53:49、55:17、1:56:44 都明說 115 年度，"
         "本場另一則筆記與散文也寫 115 年度。參考逐字稿全篇為 115年度。", "note"),
        ("25:04", "NUMBER ERROR", "1995年（民國114年）", "1995年（民國115年）", "同上。", "note"),
        ("26:03", "NUMBER ERROR", "1995年（民國114年）", "1995年（民國115年）", "同上。", "note"),
        ("26:24", "NUMBER ERROR", "1995年（民國114年）", "1995年（民國115年）", "同上。", "note"),
        (1, "NUMBER ERROR", "繼續審查民國114年總預算", "繼續審查民國115年總預算",
         "與同場筆記、散文及逐字稿 53:49 等處一致。", "summary"),
        (4, "CITATION DRIFT", "[39:27][1:53:51][1:24:28]", "[39:27][1:51:48][1:24:28]",
         "1:53:51 講的是審計部各案，與本點的主計總處無關；第27案的決議在 1:51:48，散文已引用該處。",
         "summary"),
    ],
    "ivod_17491": [(
        None, "MISATTRIBUTION",
        "由翁昭祺委員率同與會人員出席說明", "由莊昭吉委員率同與會人員出席說明",
        "2:02:19 逐字稿為「由莊昭吉委員率眾來做說明」；翁昭祺是本場主席。筆記與編號摘要都寫對，只有散文換成了主席的名字。",
        "prose",
    )],
    "ivod_17595": [(
        4, "CITATION DRIFT", "[2:04:22]", "[2:03:53][2:04:04][2:04:22]",
        "2:04:22 講的是財政部印刷廠；本點開頭的土地銀行第1、2案與凍結1000萬在 2:03:53 與 2:04:04。",
        "summary",
    )],
    "ivod_17627": [(
        None, "MISATTRIBUTION",
        "教育部代表於[2:01:21]表示毛髮可作為複驗手段，贊成多一項選項",
        "教育部代表於[1:59:24]對毛髮篩檢表示仍須與專業單位討論可行性",
        "2:01:21 是委員補充說明教育部同仁沒講到的部分，不是教育部的發言；教育部在 1:59:24 的說法是保留的，"
        "並未表示贊成。",
        "prose",
    )],
}

ROUNDS[8] = ROUND8


# Round 9: round 6's finding. Same class as round 8's year error, one step subtler -- here the
# note hedged its guess ("應為...語音辨識誤") and the hedge was lost on the way to the summary,
# so an ambiguous reading arrived as confident fact. Where the transcript cannot settle a
# detail, the repair drops the detail instead of choosing between two unsupported readings.
ROUND9 = {
    "ivod_16376": [
        ("1:22:14", "UNSUPPORTED INFERENCE",
         "第十二次會議紀錄（2020年5月13日，應為2024年5月13日，語音辨識誤）",
         "第十二次會議紀錄（逐字稿日期辨識不清，同時出現「2020年5月13日」與「10月10日星期二」兩種讀法）",
         "1:22:14 一行內同時有「2020 年 5 月 13 日」與「10 月 10 日星期二」，並非單一可修的誤讀；"
         "筆記只改年份卻保留 5月13日，等於在兩個都無依據的讀法中挑一個。參考逐字稿讀為「12月10日星期二」，"
         "同場 1:23:31 另一場會議為 113年12月20日，均不支援五月。",
         "note"),
        (1, "UNSUPPORTED INFERENCE",
         "第十二次會議紀錄（2024年5月13日），地點華茂201會議室",
         "第十二次會議紀錄（逐字稿日期辨識不清），地點華茂201會議室",
         "筆記原有的「應為…語音辨識誤」在寫進摘要時被拿掉，一個存疑的讀法變成了確定的日期。",
         "summary"),
    ],
}

ROUNDS[9] = ROUND9


# Round 10: round 6's second batch. One repair, one finding overturned on checking.
ROUND10 = {
    "ivod_17478": [
        (2, "UNSUPPORTED INFERENCE",
         "楊委員提案252萬由凍結改為主決議",
         "有委員提案252萬由凍結改為主決議",
         "筆記把發言者記為「楊委員（推測）」並標明是推測，逐字稿只標 S3，沒有任何線索指向楊委員；"
         "推測在寫進摘要時失去了但書，變成確定的歸屬。",
         "summary"),
        (None, "UNSUPPORTED INFERENCE",
         "楊委員提案將252萬由凍結改為主決議",
         "有委員提案將252萬由凍結改為主決議",
         "同上，散文也沿用了未標示的推測。",
         "prose"),
    ],
}

ROUNDS[10] = ROUND10


# Round 11: round 7's findings. Three of the five were introduced by my own earlier repairs --
# a repair is an edit like any other and can add a defect while removing one.
ROUND11 = {
    "ivod_14064": [(
        None, "UNSUPPORTED INFERENCE",
        "臺灣在亞洲國家中是少數未落實十八歲公民權者",
        "臺灣是亞洲唯一未落實十八歲公民權的民主國家",
        "第四輪修正發言者歸屬時，順手把「唯一」改成了「少數」，但逐字稿 38:40「臺灣是目前亞洲唯一的一個」"
        "與 1:07:22 都說唯一，編號摘要也仍寫唯一。修正歸屬不該連帶更動內容。",
        "prose",
    )],
    "ivod_16479": [(
        2, "CITATION DRIFT", "[38:22][38:43]", "[34:28][38:22][38:43]",
        "本點含兩段：88.26% 的整體執行率與 35.47% 的一般建築。第一輪把出處改到 35.47% 所在的 38:22-38:43 後，"
        "88.26% 那段就沒有出處了；它在 34:28 的同一段報告內。",
        "summary",
    )],
    "ivod_17211": [(
        "26:36", "UNSUPPORTED INFERENCE",
        "地點紅樓（三樓）一會議室", "地點紅樓301會議室",
        "逐字稿為「紅樓山林一會議室」，筆記把它拆成樓層加房號；參考逐字稿為「紅樓三零一會議室」，"
        "是房號301，不是三樓的第一會議室。",
        "note",
    )],
    "ivod_17242": [
        ("28:06", "MISATTRIBUTION",
         "邀請僑務委員會副委員長徐家清（代理委員長許家慶出國）報告",
         "邀請僑務委員會委員長徐家清報告，因其公務出國，由副委員長李延會代理",
         "28:06 稱徐家清為委員長，28:21 介紹「僑務委員會副委員長李延會」，30:12 說明「委員長許家慶現在公務出國，"
         "今天由我來代理報告」——許家慶與徐家清是同一個名字的兩種辨識。筆記把缺席的委員長寫成了副委員長，"
         "於是摘要出現兩個不同的人都被稱為代理報告的副委員長。",
         "note"),
        (1, "MISATTRIBUTION",
         "邀請僑委會副委員長徐家清代理委員長報告",
         "邀請僑委會委員長徐家清報告，因公務出國由副委員長李延會代理",
         "同上；本場第2點已正確寫李延會。",
         "summary"),
    ],
    "ivod_17258": [(
        1, "NUMBER ERROR", "一致推舉盧先義為招委", "一致推舉盧憲一為招委",
        "本點引用 1:11:25，該處與 1:12:44 逐字稿都作「盧憲一」；「盧先義」是同一人在 1:35:14 的另一種辨識。"
        "引用哪一行，就用那一行的寫法。第4點引用 1:35:14 起，維持盧先義不變。",
        "summary",
    )],
}

ROUNDS[11] = ROUND11


# Round 12: the audit of round 11 found one of my own repairs incomplete. I had added 34:28 to
# cover the 88.26% clause, but 34:28 is the report's opening sentence; the figure is at 35:47.
# Citing the neighbourhood of a fact is not citing the fact.
ROUND12 = {
    "ivod_16479": [(
        2, "CITATION DRIFT", "[34:28][38:22][38:43]", "[35:47][38:22][38:43]",
        "34:28 是報告開場白，88.26% 在 35:47（「累計實支數32億3702萬5000元，執行率88.26%」）。"
        "上一輪補的出處只是挨著事實，不是事實本身。",
        "summary",
    )],
}

ROUNDS[12] = ROUND12


# Round 13: round 8's remaining two, which are ordinary defects rather than derivability leaks.
ROUND13 = {
    "ivod_17452": [
        ("1:21:44", "MISATTRIBUTION", "\u5f35\u59d4\u54e1", "\u6c6a\u59d4\u54e1",
         "40人4400萬的質疑出自 S4，主席後來述明為「汪委員特別講到」；"
         "張委員只說「我跟汪委員的看法都是一致的」。", "note"),
    ],
    "ivod_17627": [
        (4, "NUMBER ERROR", "毛髮可追溯數月", "毛髮可追溯數年",
         "50:18 逐字稿為「檢驗出的時間可以長達數年」，筆記也寫數年，"
         "只有編號摘要寫成數月。", "summary"),
    ],
}

ROUNDS[13] = ROUND13


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=GOLD)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--round", type=int, default=2,
                    help="1 and 2 are review rounds; 3 repairs the notes those rounds implicated")
    args = ap.parse_args()

    for sid, items in sorted(ROUNDS[args.round].items()):
        path = os.path.join(args.gold, sid + ".json")
        doc = json.load(open(path, encoding="utf-8"))
        log = doc.setdefault("repairs", [])
        for item in items:
            point_no, kind, old, new, why = item[:5]
            surface = item[5] if len(item) > 5 else "summary"
            if surface in ("note", "note-ts"):
                target = next((n for n in doc["notes"] if n["ts"] == point_no), None)
                if target is None:
                    print("SKIP %s note %s: no note at that timestamp" % (sid, point_no))
                    continue
                point = target["ts"] if surface == "note-ts" else target["text"]
            elif surface == "prose":
                point = doc["prose"]
            else:
                point = doc["summary"][point_no - 1]
            if old is not None and old not in point:
                print("SKIP %s %s %s: fragment not found" % (sid, surface, point_no))
                continue
            fixed = new if old is None else point.replace(old, new)
            print("%s %s %s [%s]\n  - %s\n  + %s" % (sid, surface, point_no, kind, point, fixed))
            if args.dry_run:
                continue
            if surface == "note-ts":
                target["ts"] = fixed
            elif surface == "note":
                target["text"] = fixed
            elif surface == "prose":
                doc["prose"] = fixed
            else:
                doc["summary"][point_no - 1] = fixed
            log.append({
                "stage": "semantic-review-r%d" % args.round,
                "surface": surface,
                "point": point_no,
                "kind": kind,
                "before": point,
                "after": fixed,
                "why": why,
            })
        if not args.dry_run:
            json.dump(doc, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
