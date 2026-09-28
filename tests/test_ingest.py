import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.ingest import (STREAMING_1P5B, Line, chunks_to_transcript, clean_text,  # noqa: E402
                               format_ts, parse_line, resolve_citation, segments_to_lines)

IDENTITY = lambda s: s  # noqa: E731


class IngestTest(unittest.TestCase):
    def test_geometry(self):
        self.assertAlmostEqual(STREAMING_1P5B.chunk_seconds, 2.9333, places=4)

    def test_format_ts(self):
        self.assertEqual(format_ts(8), "0:08")
        self.assertEqual(format_ts(3761), "1:02:41")

    def test_clean_text_keeps_meaningful_words(self):
        self.assertEqual(clean_text("嗯，那個預算[Noise]就是這樣"), "那個預算就是這樣")
        self.assertEqual(clean_text("[Laughter] [Noise]"), "")
        self.assertEqual(clean_text("標題是「[草案]」"), "標題是「[草案]」")

    def test_speaker_renumbering_and_unknown(self):
        segs = [
            {"Start": 0.0, "End": 12.0, "Speaker": "?", "Content": "開始"},
            {"Start": 12.0, "End": 25.0, "Speaker": "3", "Content": "第一位"},
            {"Start": 25.0, "End": 40.0, "Speaker": "0", "Content": "第二位"},
            {"Start": 40.0, "End": 52.0, "Speaker": "3", "Content": "又是第一位"},
            {"Start": 52.0, "End": 60.0, "Speaker": "0", "Content": "[Noise]"},
        ]
        lines = segments_to_lines(segs, IDENTITY)
        self.assertEqual([l.render() for l in lines],
                         ["[0:00] 開始", "[0:12] S1: 第一位", "[0:25] S2: 第二位", "[0:40] S1: 又是第一位"])

    def test_chunks_carry_speaker_forward(self):
        chunks = ["Speaker 0: 大家好，今天討論預算。", "我們先看第一案。", "Speaker 1: 好的。"]
        text = chunks_to_transcript(chunks, convert=IDENTITY)
        rendered = text.splitlines()
        self.assertTrue(rendered[0].startswith("[0:00] S1: 大家好"))
        self.assertIn("第一案", rendered[0])
        self.assertTrue(rendered[-1].endswith("S2: 好的。"))

    def test_parse_round_trip(self):
        for line in [Line(8, "S2", "上次提的採購案"), Line(3761, None, "無講者"),
                     Line(65, "S1", "時間 [1:02] 與冒號: 都在內文")]:
            self.assertEqual(parse_line(line.render()), line)

    def test_resolve_citation(self):
        lines = [Line(8, "S1", "a"), Line(8, "S2", "b"), Line(20, "S1", "c")]
        self.assertEqual(resolve_citation("0:08", lines), 0)
        self.assertEqual(resolve_citation("0:20", lines), 2)
        self.assertIsNone(resolve_citation("0:09", lines))
        self.assertIsNone(resolve_citation("abc", lines))


if __name__ == "__main__":
    unittest.main()
