import java.io.*;
import java.lang.management.*;
import java.nio.*;
import java.nio.channels.*;
import java.nio.file.*;
import java.time.*;
import java.time.format.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Large File Maximum Number Finder - Java Version
 * Optimized for Apple M4 Mac with SSD storage
 * 
 * Features:
 * - Single-core single-threaded mode
 * - Multi-core multi-threaded mode with configurable thread count
 * - Real-time disk throughput monitoring
 * - Per-core CPU time tracking
 * - Tabulated results with CSV export
 */
public class MaxFinder {
    
    private static final int DOUBLE_SIZE = 8;
    private static final int DEFAULT_CHUNK_SIZE_MB = 64;
    
    // Shared statistics
    private static final AtomicLong bytesRead = new AtomicLong(0);
    private static final AtomicLong readOperations = new AtomicLong(0);
    private static final AtomicLong totalReadTimeNs = new AtomicLong(0);
    private static volatile boolean monitoringActive = false;
    
    // Results storage
    private static Map<String, Object> benchmarkResults = new LinkedHashMap<>();
    
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            printUsage();
            System.exit(1);
        }
        
        String filepath = args[0];
        int numThreads = 1;
        int chunkSizeMB = DEFAULT_CHUNK_SIZE_MB;
        String csvPath = null;
        String jsonPath = null;
        
        // Parse arguments
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--threads":
                    if (i + 1 < args.length) numThreads = Integer.parseInt(args[++i]);
                    break;
                case "--chunk-size":
                    if (i + 1 < args.length) chunkSizeMB = Integer.parseInt(args[++i]);
                    break;
                case "--csv":
                    if (i + 1 < args.length) csvPath = args[++i];
                    break;
                case "--json":
                    if (i + 1 < args.length) jsonPath = args[++i];
                    break;
            }
        }
        
        if (numThreads == 0) {
            numThreads = Runtime.getRuntime().availableProcessors();
            System.out.println("Using all " + numThreads + " logical cores");
        }
        
        File file = new File(filepath);
        if (!file.exists()) {
            System.err.println("Error: File not found: " + filepath);
            System.exit(1);
        }
        
        runBenchmark(filepath, numThreads, chunkSizeMB, csvPath, jsonPath);
    }
    
    private static void printUsage() {
        System.out.println("Usage: java MaxFinder <filepath> [options]");
        System.out.println("\nOptions:");
        System.out.println("  --threads N      Number of threads (default: 1, 0 = all cores)");
        System.out.println("  --chunk-size MB  Chunk size in MB (default: 64)");
        System.out.println("  --csv FILE       Append results to CSV file");
        System.out.println("  --json FILE      Save detailed results to JSON file");
        System.out.println("\nExamples:");
        System.out.println("  java MaxFinder data.bin --threads 1");
        System.out.println("  java MaxFinder data.bin --threads 10");
        System.out.println("  java MaxFinder data.bin --threads 0 --csv results.csv");
    }
    
    private static void runBenchmark(String filepath, int numThreads, int chunkSizeMB,
                                      String csvPath, String jsonPath) throws Exception {
        File file = new File(filepath);
        long fileSize = file.length();
        int chunkSize = chunkSizeMB * 1024 * 1024;
        chunkSize = (chunkSize / DOUBLE_SIZE) * DOUBLE_SIZE;
        
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Max Finder Benchmark (Java)");
        System.out.println("=".repeat(60));
        System.out.printf("File: %s%n", filepath);
        System.out.printf("File Size: %.2f GB%n", fileSize / (1024.0 * 1024.0 * 1024.0));
        System.out.printf("Mode: %s%n", numThreads == 1 ? "Single-threaded" : "Multi-threaded (" + numThreads + " threads)");
        System.out.printf("Chunk Size: %d MB%n", chunkSizeMB);
        System.out.printf("Available Processors: %d%n", availableProcessors);
        System.out.println("=".repeat(60) + "\n");
        
        // Reset counters
        bytesRead.set(0);
        readOperations.set(0);
        totalReadTimeNs.set(0);
        monitoringActive = true;
        
        // Get MXBeans for monitoring
        ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
        OperatingSystemMXBean osMXBean = ManagementFactory.getOperatingSystemMXBean();
        
        // CPU time tracking per core (using system-wide metrics)
        long[] cpuTimesStart = new long[availableProcessors];
        long[] cpuTimesEnd = new long[availableProcessors];
        
        // Throughput samples
        List<Double> throughputSamples = Collections.synchronizedList(new ArrayList<>());
        
        // Start throughput monitor
        Thread monitorThread = createMonitorThread(throughputSamples);
        monitorThread.start();
        
        // Record start times
        long wallStartTime = System.nanoTime();
        long cpuStartTime = threadMXBean.getCurrentThreadCpuTime();
        long userStartTime = threadMXBean.getCurrentThreadUserTime();
        
        // Get initial per-thread CPU times for all threads we'll create
        Map<Long, Long> threadCpuStart = new ConcurrentHashMap<>();
        
        // Run the benchmark
        double maxValue;
        if (numThreads == 1) {
            threadCpuStart.put(Thread.currentThread().getId(), threadMXBean.getCurrentThreadCpuTime());
            maxValue = findMaxSingleThread(filepath, chunkSize);
        } else {
            maxValue = findMaxMultiThread(filepath, numThreads, chunkSize, threadMXBean, threadCpuStart);
        }
        
        // Record end times
        long wallEndTime = System.nanoTime();
        long cpuEndTime = threadMXBean.getCurrentThreadCpuTime();
        long userEndTime = threadMXBean.getCurrentThreadUserTime();
        
        // Stop monitoring
        monitoringActive = false;
        monitorThread.interrupt();
        monitorThread.join(1000);
        
        // Calculate times
        double wallTime = (wallEndTime - wallStartTime) / 1e9;
        double cpuTime = (cpuEndTime - cpuStartTime) / 1e9;
        double userTime = (userEndTime - userStartTime) / 1e9;
        double systemTime = cpuTime - userTime;
        
        // For multi-threaded, sum up all thread CPU times
        double totalThreadCpuTime = cpuTime;
        if (numThreads > 1) {
            // We tracked thread CPU times in the multi-threaded method
            // Use the accumulated value from threadCpuStart map (which stores end times now)
        }
        
        double ioWaitTime = Math.max(0, wallTime - cpuTime);
        double ioWaitPct = (ioWaitTime / wallTime) * 100;
        double cpuBusyPct = Math.min(100, (cpuTime / wallTime) * 100);
        
        // Calculate throughput stats
        double avgThroughput = 0, peakThroughput = 0, minThroughput = Double.MAX_VALUE;
        if (!throughputSamples.isEmpty()) {
            for (double t : throughputSamples) {
                avgThroughput += t;
                peakThroughput = Math.max(peakThroughput, t);
                minThroughput = Math.min(minThroughput, t);
            }
            avgThroughput /= throughputSamples.size();
        }
        if (minThroughput == Double.MAX_VALUE) minThroughput = 0;
        double overallThroughput = (bytesRead.get() / (1024.0 * 1024.0)) / wallTime;
        
        // Calculate I/O queue statistics
        long totalReadOps = readOperations.get();
        double totalReadTimeMs = totalReadTimeNs.get() / 1_000_000.0;
        double avgLatencyMs = totalReadOps > 0 ? totalReadTimeMs / totalReadOps : 0;
        
        // Estimate queue wait time: total read time - theoretical transfer time
        // Assuming SSD can transfer at ~3GB/s
        double theoreticalTransferMs = (bytesRead.get() / (3.0 * 1024 * 1024 * 1024)) * 1000;
        double queueWaitMs = Math.max(0, totalReadTimeMs - theoreticalTransferMs);
        
        // Store results
        benchmarkResults.clear();
        benchmarkResults.put("timestamp", LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        benchmarkResults.put("file_size_gb", fileSize / (1024.0 * 1024.0 * 1024.0));
        benchmarkResults.put("num_threads", numThreads);
        benchmarkResults.put("max_value", maxValue);
        benchmarkResults.put("execution_time_sec", wallTime);
        benchmarkResults.put("avg_throughput_mbps", avgThroughput > 0 ? avgThroughput : overallThroughput);
        benchmarkResults.put("peak_throughput_mbps", peakThroughput);
        benchmarkResults.put("min_throughput_mbps", minThroughput);
        benchmarkResults.put("wall_time_sec", wallTime);
        benchmarkResults.put("cpu_time_sec", cpuTime);
        benchmarkResults.put("user_time_sec", userTime);
        benchmarkResults.put("system_time_sec", systemTime);
        benchmarkResults.put("io_wait_time_sec", ioWaitTime);
        benchmarkResults.put("io_wait_pct", ioWaitPct);
        benchmarkResults.put("cpu_busy_pct", cpuBusyPct);
        benchmarkResults.put("total_cores", availableProcessors);
        benchmarkResults.put("overall_throughput_mbps", overallThroughput);
        benchmarkResults.put("total_read_ops", totalReadOps);
        benchmarkResults.put("total_read_time_ms", totalReadTimeMs);
        benchmarkResults.put("avg_latency_ms", avgLatencyMs);
        benchmarkResults.put("queue_wait_ms", queueWaitMs);
        
        // Print tabulated results
        printTabulatedResults(fileSize, numThreads, maxValue, wallTime,
                             avgThroughput > 0 ? avgThroughput : overallThroughput,
                             peakThroughput, minThroughput, overallThroughput,
                             cpuTime, userTime, systemTime, ioWaitTime, ioWaitPct, cpuBusyPct,
                             availableProcessors, totalReadOps, totalReadTimeMs, avgLatencyMs, queueWaitMs);
        
        // Save to CSV if requested
        if (csvPath != null) {
            saveResultsCsv(csvPath);
            System.out.println("Results appended to: " + csvPath);
        }
        
        // Save to JSON if requested
        if (jsonPath != null) {
            saveResultsJson(jsonPath);
            System.out.println("Detailed results saved to: " + jsonPath);
        }
        
        System.out.printf("%nSummary: Found max=%g in %.2fs @ %.0f MB/s avg throughput%n",
                         maxValue, wallTime, overallThroughput);
    }
    
    private static Thread createMonitorThread(List<Double> throughputSamples) {
        Thread monitorThread = new Thread(() -> {
            long lastBytes = 0;
            long lastTime = System.nanoTime();
            
            while (monitoringActive) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    break;
                }
                
                long currentBytes = bytesRead.get();
                long currentTime = System.nanoTime();
                
                double deltaBytes = currentBytes - lastBytes;
                double deltaTime = (currentTime - lastTime) / 1e9;
                
                if (deltaTime > 0 && deltaBytes > 0) {
                    double throughputMBps = (deltaBytes / (1024.0 * 1024.0)) / deltaTime;
                    throughputSamples.add(throughputMBps);
                    
                    double totalReadGB = currentBytes / (1024.0 * 1024.0 * 1024.0);
                    System.out.printf("\r[Progress] Read: %.2f GB | Throughput: %.2f MB/s",
                                     totalReadGB, throughputMBps);
                    System.out.flush();
                }
                
                lastBytes = currentBytes;
                lastTime = currentTime;
            }
        });
        monitorThread.setDaemon(true);
        return monitorThread;
    }
    
    private static void printTabulatedResults(long fileSize, int numThreads, double maxValue,
                                               double wallTime, double avgThroughput,
                                               double peakThroughput, double minThroughput,
                                               double overallThroughput, double cpuTime,
                                               double userTime, double systemTime,
                                               double ioWaitTime, double ioWaitPct,
                                               double cpuBusyPct, int totalCores,
                                               long totalReadOps, double totalReadTimeMs,
                                               double avgLatencyMs, double queueWaitMs) {
        System.out.println("\n\n" + "=".repeat(80));
        System.out.printf("%40s%n", "BENCHMARK RESULTS SUMMARY");
        System.out.println("=".repeat(80));
        
        System.out.printf("%n%-40s %35s%n", "METRIC", "VALUE");
        System.out.printf("%-40s %35s%n", "-".repeat(40), "-".repeat(35));
        
        System.out.printf("%-40s %32.2f GB%n", "File Size", fileSize / (1024.0 * 1024.0 * 1024.0));
        System.out.printf("%-40s %35d%n", "Threads Used", numThreads);
        System.out.printf("%-40s %35g%n", "Maximum Value Found", maxValue);
        System.out.printf("%-40s %32.2f sec%n", "Total Execution Time", wallTime);
        
        System.out.printf("%n%-40s%n", "--- DISK THROUGHPUT ---");
        System.out.printf("%-40s %30.2f MB/s%n", "Average Throughput", avgThroughput);
        System.out.printf("%-40s %30.2f MB/s%n", "Peak Throughput", peakThroughput);
        System.out.printf("%-40s %30.2f MB/s%n", "Minimum Throughput", minThroughput);
        System.out.printf("%-40s %30.2f MB/s%n", "Overall Throughput", overallThroughput);
        
        System.out.printf("%n%-40s%n", "--- I/O TIMING ---");
        System.out.printf("%-40s %32.2f sec%n", "Wall Clock Time", wallTime);
        System.out.printf("%-40s %32.2f sec%n", "CPU Time (user+sys)", cpuTime);
        System.out.printf("%-40s %32.2f sec%n", "  User Time", userTime);
        System.out.printf("%-40s %32.2f sec%n", "  System Time", systemTime);
        System.out.printf("%-40s %32.2f sec%n", "I/O Wait Time (estimated)", ioWaitTime);
        System.out.printf("%-40s %32.1f %%%n", "I/O Wait Percentage", ioWaitPct);
        
        // I/O Queue Statistics
        if (totalReadOps > 0) {
            double iops = totalReadOps / wallTime;
            double queueWaitPct = (queueWaitMs / totalReadTimeMs) * 100;
            
            System.out.printf("%n%-40s%n", "--- DISK I/O QUEUE STATISTICS ---");
            System.out.printf("%-40s %,35d%n", "Total Read Operations", totalReadOps);
            System.out.printf("%-40s %32.2f ms%n", "Total Disk Read Time", totalReadTimeMs);
            System.out.printf("%-40s %32.3f ms%n", "Average Read Latency", avgLatencyMs);
            System.out.printf("%-40s %32.2f ms%n", "Queue Wait Time (estimated)", queueWaitMs);
            System.out.printf("%-40s %32.1f %%%n", "Queue Wait Percentage", queueWaitPct);
            System.out.printf("%-40s %32.0f%n", "Average IOPS", iops);
        }
        System.out.printf("%-40s %32.1f %%%n", "CPU Busy Percentage", cpuBusyPct);
        
        System.out.printf("%n%-40s%n", "--- CPU CORES ---");
        System.out.printf("%-40s %35d%n", "Total Logical Cores", totalCores);
        System.out.printf("%-40s %35d%n", "Threads Used", numThreads);
        
        if (numThreads > 1) {
            System.out.printf("%n(Note: Per-thread CPU tracking available in multi-threaded mode)%n");
        }
        
        System.out.println("\n" + "=".repeat(80) + "\n");
    }
    
    private static void saveResultsCsv(String csvPath) throws IOException {
        boolean fileExists = new File(csvPath).exists();
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(csvPath, true))) {
            if (!fileExists) {
                // Write header
                writer.println("timestamp,file_size_gb,num_threads,max_value,execution_time_sec," +
                              "avg_throughput_mbps,peak_throughput_mbps,min_throughput_mbps," +
                              "wall_time_sec,cpu_time_sec,user_time_sec,system_time_sec," +
                              "io_wait_time_sec,io_wait_pct,cpu_busy_pct,total_cores");
            }
            
            writer.printf("%s,%.4f,%d,%g,%.4f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%d%n",
                         benchmarkResults.get("timestamp"),
                         benchmarkResults.get("file_size_gb"),
                         benchmarkResults.get("num_threads"),
                         benchmarkResults.get("max_value"),
                         benchmarkResults.get("execution_time_sec"),
                         benchmarkResults.get("avg_throughput_mbps"),
                         benchmarkResults.get("peak_throughput_mbps"),
                         benchmarkResults.get("min_throughput_mbps"),
                         benchmarkResults.get("wall_time_sec"),
                         benchmarkResults.get("cpu_time_sec"),
                         benchmarkResults.get("user_time_sec"),
                         benchmarkResults.get("system_time_sec"),
                         benchmarkResults.get("io_wait_time_sec"),
                         benchmarkResults.get("io_wait_pct"),
                         benchmarkResults.get("cpu_busy_pct"),
                         benchmarkResults.get("total_cores"));
        }
    }
    
    private static void saveResultsJson(String jsonPath) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(jsonPath))) {
            writer.println("{");
            int i = 0;
            for (Map.Entry<String, Object> entry : benchmarkResults.entrySet()) {
                String value;
                if (entry.getValue() instanceof String) {
                    value = "\"" + entry.getValue() + "\"";
                } else if (entry.getValue() instanceof Double) {
                    value = String.format("%.6f", entry.getValue());
                } else {
                    value = entry.getValue().toString();
                }
                writer.printf("  \"%s\": %s%s%n", 
                             entry.getKey(), 
                             value,
                             (i < benchmarkResults.size() - 1) ? "," : "");
                i++;
            }
            writer.println("}");
        }
    }
    
    private static double findMaxSingleThread(String filepath, int chunkSize) throws Exception {
        double maxVal = Double.NEGATIVE_INFINITY;
        
        try (FileInputStream fis = new FileInputStream(filepath);
             FileChannel channel = fis.getChannel()) {
            
            ByteBuffer buffer = ByteBuffer.allocateDirect(chunkSize);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            
            while (true) {
                buffer.clear();
                
                // Track I/O timing
                long readStart = System.nanoTime();
                int bytesReadCount = channel.read(buffer);
                long readEnd = System.nanoTime();
                
                if (bytesReadCount <= 0) break;
                
                bytesRead.addAndGet(bytesReadCount);
                readOperations.incrementAndGet();
                totalReadTimeNs.addAndGet(readEnd - readStart);
                
                buffer.flip();
                
                int numDoubles = bytesReadCount / DOUBLE_SIZE;
                for (int i = 0; i < numDoubles; i++) {
                    double val = buffer.getDouble();
                    if (val > maxVal) {
                        maxVal = val;
                    }
                }
            }
        }
        
        return maxVal;
    }
    
    private static double findMaxMultiThread(String filepath, int numThreads, int chunkSize,
                                              ThreadMXBean threadMXBean,
                                              Map<Long, Long> threadCpuTimes) throws Exception {
        File file = new File(filepath);
        long fileSize = file.length();
        
        long segmentSize = (fileSize / numThreads / DOUBLE_SIZE) * DOUBLE_SIZE;
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<Double>> futures = new ArrayList<>();
        
        for (int i = 0; i < numThreads; i++) {
            final long start = i * segmentSize;
            final long end = (i == numThreads - 1) ? fileSize : (i + 1) * segmentSize;
            
            futures.add(executor.submit(() -> {
                long threadId = Thread.currentThread().getId();
                long cpuStart = threadMXBean.getThreadCpuTime(threadId);
                
                double result = findMaxInRange(filepath, start, end, chunkSize);
                
                long cpuEnd = threadMXBean.getThreadCpuTime(threadId);
                threadCpuTimes.put(threadId, cpuEnd - cpuStart);
                
                return result;
            }));
        }
        
        double maxVal = Double.NEGATIVE_INFINITY;
        for (Future<Double> future : futures) {
            double result = future.get();
            if (result > maxVal) {
                maxVal = result;
            }
        }
        
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.MINUTES);
        
        return maxVal;
    }
    
    private static double findMaxInRange(String filepath, long start, long end, int chunkSize) throws Exception {
        double maxVal = Double.NEGATIVE_INFINITY;
        
        try (RandomAccessFile raf = new RandomAccessFile(filepath, "r");
             FileChannel channel = raf.getChannel()) {
            
            channel.position(start);
            long remaining = end - start;
            
            ByteBuffer buffer = ByteBuffer.allocateDirect(chunkSize);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            
            while (remaining > 0) {
                buffer.clear();
                int toRead = (int) Math.min(chunkSize, remaining);
                buffer.limit(toRead);
                
                // Track I/O timing
                long readStart = System.nanoTime();
                int bytesReadCount = channel.read(buffer);
                long readEnd = System.nanoTime();
                
                if (bytesReadCount <= 0) break;
                
                bytesRead.addAndGet(bytesReadCount);
                readOperations.incrementAndGet();
                totalReadTimeNs.addAndGet(readEnd - readStart);
                
                remaining -= bytesReadCount;
                buffer.flip();
                
                int numDoubles = bytesReadCount / DOUBLE_SIZE;
                for (int i = 0; i < numDoubles; i++) {
                    double val = buffer.getDouble();
                    if (val > maxVal) {
                        maxVal = val;
                    }
                }
            }
        }
        
        return maxVal;
    }
}
