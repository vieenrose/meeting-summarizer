import com.google.ai.edge.litertlm.*;
import java.nio.file.*;
import java.util.*;

public class Bench {
  static long rssKb(String key) throws Exception {
    for (String l : Files.readAllLines(Paths.get("/proc/self/status")))
      if (l.startsWith(key)) return Long.parseLong(l.replaceAll("[^0-9]", ""));
    return -1;
  }
  static String mem() throws Exception {
    StringBuilder b = new StringBuilder();
    for (String l : Files.readAllLines(Paths.get("/proc/self/smaps_rollup")))
      for (String k : new String[]{"Rss:", "Anonymous:", "Pss_Anon:", "Pss_File:", "Pss_Shmem:", "Private_Dirty:"})
        if (l.startsWith(k)) b.append(k).append(Long.parseLong(l.replaceAll("[^0-9]", "")) / 1024).append("MB ");
    b.append("HWM:").append(rssKb("VmHWM") / 1024).append("MB");
    return b.toString();
  }
  public static void main(String[] a) throws Exception {
    String model = a[0];
    int threads = Integer.parseInt(a[1]);
    int maxTokens = Integer.parseInt(a[2]);
    String[] parts = new String(Files.readAllBytes(Paths.get(a[3])), "UTF-8").split("\u001e");
    int nChunks = Integer.parseInt(a[4]);
    int decodeTurns = a.length > 5 ? Integer.parseInt(a[5]) : 1;
    if (System.getenv("SPEC") != null) ExperimentalFlags.INSTANCE.setEnableSpeculativeDecoding(Boolean.valueOf(System.getenv("SPEC").equals("1")));
    System.out.println("speculative " + ExperimentalFlags.INSTANCE.getEnableSpeculativeDecoding());
    long t0 = System.nanoTime();
    Backend be = threads == 0 ? (Backend) new Backend.GPU() : (Backend) new Backend.CPU(threads, null);
    EngineConfig cfg = new EngineConfig(model, be, null, null, maxTokens, null, "/data/local/tmp/lt/cache");
    Engine eng = new Engine(cfg);
    eng.initialize();
    System.out.printf("init %.1fs rss %d MB hwm %d MB%n", (System.nanoTime() - t0) / 1e9, rssKb("VmRSS") / 1024, rssKb("VmHWM") / 1024);
    System.out.println("MEM after-init " + mem());
    Session s = eng.createSession(new SessionConfig());
    int done = 0, chunks = 0;
    long tp = 0;
    for (int ci = 0; ci < Math.min(nChunks, parts.length); ci++) {
      String c = parts[ci];
      long t = System.nanoTime();
      s.runPrefill(Collections.singletonList(new InputData.Text(c)));
      long dt = System.nanoTime() - t; tp += dt;
      done += c.length(); chunks++;
      System.out.printf("prefill chunk %d chars %d total %d in %.2fs (%.0f chars/s) rss %d MB%n", chunks, c.length(), done, dt / 1e9, c.length() / (dt / 1e9), rssKb("VmRSS") / 1024);
    }
    System.out.println("MEM after-prefill " + mem());
    final int capSec = a.length > 6 ? Integer.parseInt(a[6]) : 0;
    for (int i = 0; i < decodeTurns; i++) {
      long t = System.nanoTime();
      if (capSec > 0) { final Session ss = s; Thread th = new Thread(() -> { try { Thread.sleep(capSec * 1000L); ss.cancelProcess(); } catch (Exception e) {} }); th.setDaemon(true); th.start(); }
      String out;
      try { out = s.runDecode(); } catch (Exception e) { out = "<cancelled: " + e.getMessage() + ">"; }
      long dt = System.nanoTime() - t;
      Files.write(Paths.get("/data/local/tmp/lt/decode_out.txt"), out.getBytes("UTF-8"));
      System.out.printf("decode %d chars in %.2fs: %s%n", out.length(), dt / 1e9, out.replace("\n", " ").substring(0, Math.min(120, out.length())));
    }
    System.out.printf("peak hwm %d MB%n", rssKb("VmHWM") / 1024);
    System.out.println("MEM after-decode " + mem());
    s.close(); eng.close();
  }
}
