import com.google.ai.edge.litertlm.*;
import java.nio.file.*;
import java.util.*;

// Reading-agent protocol on LiteRT-LM: incremental prefill of transcript chunks, a decode per window,
// two ways to join turns after runDecode, and a one-shot reference. Prints per-chunk prefill speed and memory.
public class Reader {
  static long kb(String key) throws Exception {
    for (String l : Files.readAllLines(Paths.get("/proc/self/status"))) if (l.startsWith(key)) return Long.parseLong(l.replaceAll("[^0-9]", ""));
    return -1;
  }
  static void pre(Session s, String t, String tag) throws Exception {
    long t0 = System.nanoTime();
    s.runPrefill(Collections.singletonList(new InputData.Text(t)));
    double dt = (System.nanoTime() - t0) / 1e9;
    System.out.printf("prefill %s chars %d in %.2fs (%.0f ch/s) rss %d MB%n", tag, t.length(), dt, t.length() / dt, kb("VmRSS") / 1024);
  }
  static String dec(Session s, String tag) throws Exception {
    long t0 = System.nanoTime();
    String o = s.runDecode();
    double dt = (System.nanoTime() - t0) / 1e9;
    System.out.printf("decode %s chars %d in %.2fs (%.1f ch/s) rss %d MB hwm %d MB%n", tag, o.length(), dt, o.length() / dt, kb("VmRSS") / 1024, kb("VmHWM") / 1024);
    return o;
  }
  public static void main(String[] a) throws Exception {
    String[] g = new String(Files.readAllBytes(Paths.get(a[2])), "UTF-8").split("\u001d");
    String prefix = g[0];
    String[] w1 = g[1].split("\u001e"), w2 = g[2].split("\u001e");
    String w2head = g[3];
    int threads = Integer.parseInt(a[1]);
    if (System.getenv("SPEC") != null) ExperimentalFlags.INSTANCE.setEnableSpeculativeDecoding(System.getenv("SPEC").equals("1"));
    Backend be = threads == 0 ? (Backend) new Backend.GPU() : (Backend) new Backend.CPU(threads, null);
    long t0 = System.nanoTime();
    Engine eng = new Engine(new EngineConfig(a[0], be, null, null, 8192, null, "/data/local/tmp/lt/cache"));
    eng.initialize();
    System.out.printf("init %.1fs rss %d MB%n", (System.nanoTime() - t0) / 1e9, kb("VmRSS") / 1024);
    SessionConfig sc = new SessionConfig(new SamplerConfig(1, 1.0, 0.0, 0), null);
    StringBuilder out = new StringBuilder();
    String o1 = null;
    for (String join : new String[]{"<turn|>\n", ""}) {
      Session s = eng.createSession(sc);
      pre(s, prefix, "prefix");
      for (int i = 0; i < w1.length; i++) pre(s, w1[i], "w1." + i);
      pre(s, "<turn|>\n<|turn>model\n", "close");
      String r1 = dec(s, "w1");
      if (o1 == null) o1 = r1;
      pre(s, join + "<|turn>user\n" + w2head, "open2");
      for (int i = 0; i < w2.length; i++) pre(s, w2[i], "w2." + i);
      pre(s, "<turn|>\n<|turn>model\n", "close");
      String r2 = dec(s, "w2");
      out.append("=== join ").append(join.isEmpty() ? "none" : "turn").append("\nW1:\n").append(r1).append("\nW2:\n").append(r2).append("\n");
      s.close();
    }
    Session s = eng.createSession(sc);
    StringBuilder full = new StringBuilder(prefix);
    for (String c : w1) full.append(c);
    full.append("<turn|>\n<|turn>model\n").append(o1).append("<turn|>\n<|turn>user\n").append(w2head);
    for (String c : w2) full.append(c);
    full.append("<turn|>\n<|turn>model\n");
    pre(s, full.toString(), "reference");
    out.append("=== reference\nW2:\n").append(dec(s, "ref")).append("\n");
    s.close(); eng.close();
    Files.write(Paths.get("/data/local/tmp/lt/reader_out.txt"), out.toString().getBytes("UTF-8"));
    System.out.printf("peak hwm %d MB%n", kb("VmHWM") / 1024);
  }
}
