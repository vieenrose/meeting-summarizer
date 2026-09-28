import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.ingest import Line  # noqa: E402
from summarizer.pipeline import (NotesPipeline, PipelineConfig, make_windows, parse_turn)  # noqa: E402

TRANSCRIPT = "\n".join([
    "[0:00] S1: 今天審查能源管理法，請各位委員發言。",
    "[0:20] S2: 行政院版第二條沒有意見，建議照案通過。",
    "[5:00] S1: 第三條保留，下次再議，請經濟部補充說明。",
    "[5:30] S3: 預算刪減一億元，這個數字必須記錄清楚。",
    "[40:00] S1: 第五條照案通過，沒有異議。",
    "[41:00] S2: 請經濟部兩週內提出書面報告。",
])


def count(text):
    return len(text)


class ParseTurnTest(unittest.TestCase):
    def test_accepts_every_observed_shape(self):
        for line in ["- [5:00] 主席: 第三條保留",
                     "NOTE: [5:00] 主席: 第三條保留",
                     "NOTE：[5:00] 主席: 第三條保留",
                     "1. [5:00] 主席: 第三條保留",
                     "[5:00] 主席: 第三條保留",
                     "- (DECISION) [5:00] 主席: 第三條保留",
                     "- （DECISION）［5:00］主席: 第三條保留"]:
            turn = parse_turn(line)
            self.assertEqual(len(turn.notes), 1, line)
            self.assertEqual(turn.notes[0][1], "5:00", line)
            self.assertEqual(turn.rejected, [], line)

    def test_nothing_new_and_unknown_lines(self):
        turn = parse_turn("NOTHING-NEW：只有點名\n閒聊一句")
        self.assertEqual(turn.nothing_new, "只有點名")
        self.assertEqual(turn.rejected, ["閒聊一句"])
        self.assertEqual(turn.notes, [])


class WindowsTest(unittest.TestCase):
    def test_in_order_packing_keeps_every_line(self):
        lines = [Line(i * 10, "S1", "字" * 30) for i in range(10)]
        windows = make_windows(lines, 100, count)
        self.assertEqual([l.start_s for w in windows for l in w], [i * 10 for i in range(10)])


class PipelineTest(unittest.TestCase):
    def test_reads_every_window_and_writes_cited_summary(self):
        class Chat:
            def __init__(self):
                self.window_prompts = []

            def __call__(self, messages):
                last = messages[-1]["content"]
                if last.startswith("以下是整場會議"):
                    return ("1. 第二條照案通過 [0:20]\n2. 第三條保留 [5:00]\n"
                            "3. 第五條通過，經濟部兩週內報告 [40:00] [41:00]")
                self.window_prompts.append(last)
                ts = last.split("[")[1].split("]")[0]
                return f"- [{ts}] 主席: 這段的決議"

        chat = Chat()
        result = NotesPipeline(chat, count, PipelineConfig(window_tokens=60)).run(TRANSCRIPT)
        self.assertGreaterEqual(result["windows"], 3)
        self.assertEqual(result["windows_without_notes"], [])
        self.assertEqual(len(result["summary"]), 3)
        self.assertEqual([l for l in result["log"] if l.get("step") == "write"][-1]["problems"], [])
        # Windows are independent: no window prompt may carry earlier notes.
        self.assertFalse(any("note 1:" in p for p in chat.window_prompts))
        # And no window may be told it can stop.
        self.assertFalse(any("DONE" in p for p in chat.window_prompts))

    def test_invented_timestamp_rejected_out_of_window_kept(self):
        pipeline = NotesPipeline(lambda m: "", count, PipelineConfig())
        lines = [parse for parse in [Line(0, "S1", "a"), Line(300, "S1", "b")]]
        notes = []
        turn = parse_turn("- [0:00] 主席: 真實\n- [5:00] 主席: 先前視窗\n- [9:99] 主席: 編造")
        added = pipeline._accept(turn, 2, [lines[0]], notes, lines)
        self.assertEqual(added, 2)
        self.assertTrue(any("invented" in r for r in turn.rejected))
        self.assertTrue(any("out-of-window" in r for r in turn.rejected))

    def test_window_with_speech_must_produce_notes(self):
        long_line = "委員質詢預算刪減一億元，主席裁示照案通過。" * 12

        class Chat:
            def __init__(self):
                self.tries = 0

            def __call__(self, messages):
                if messages[-1]["content"].startswith("以下是整場會議"):
                    return "1. 預算案通過 [0:00]"
                self.tries += 1
                return "NOTHING-NEW: 沒有新資訊" if self.tries == 1 else "- [0:00] 主席: 預算照案通過"

        result = NotesPipeline(Chat(), count, PipelineConfig(window_tokens=10_000)).run(
            f"[0:00] S1: {long_line}")
        self.assertEqual(result["windows_without_notes"], [])
        self.assertTrue(any(l.get("retry") == "no-notes" for l in result["log"]))

    def test_quiet_window_may_be_skipped(self):
        class Chat:
            def __call__(self, messages):
                if messages[-1]["content"].startswith("以下是整場會議"):
                    return "1. 無實質內容 [0:00]"
                return "NOTHING-NEW: 只有靜音"

        result = NotesPipeline(Chat(), count, PipelineConfig(window_tokens=10_000)).run("[0:00] S1: 好")
        self.assertEqual(result["notes"], [])
        self.assertFalse(any(l.get("retry") for l in result["log"]))


if __name__ == "__main__":
    unittest.main()
