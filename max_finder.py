#!/usr/bin/env python3
"""
Large File Maximum Number Finder
Optimized for Apple M4 Mac with SSD storage

Supports:
- Single-core single-threaded mode
- Multi-core multi-threaded mode with configurable thread count
- Real-time disk throughput and CPU utilization monitoring
- Per-core CPU time tracking (on-CPU vs off-CPU)
- Tabulated results with CSV export
"""

import argparse
import csv
import json
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict

import psutil


@dataclass
class Stats:
    """Statistics for monitoring performance."""
    bytes_read: int = 0
    start_time: float = 0.0
    cpu_time_start: float = 0.0
    lock: threading.Lock = None
    
    def __post_init__(self):
        if self.lock is None:
            self.lock = threading.Lock()
    
    def add_bytes(self, count: int):
        with self.lock:
            self.bytes_read += count
    
    def get_throughput_mbps(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return (self.bytes_read / (1024 * 1024)) / elapsed
        return 0.0


class IOWaitMonitor:
    """
    Monitor I/O wait time by tracking wall time vs CPU time.
    
    On macOS, true iowait isn't exposed, so we calculate:
    - Wall time: Total elapsed time
    - CPU time: Time actually spent executing on CPU
    - I/O wait (estimated): Wall time - CPU time
    """
    
    def __init__(self):
        self.wall_start = 0.0
        self.cpu_start = 0.0
        self.process = psutil.Process()
        
    def start(self):
        self.wall_start = time.time()
        cpu_times = self.process.cpu_times()
        self.cpu_start = cpu_times.user + cpu_times.system
        
    def get_stats(self) -> dict:
        wall_elapsed = time.time() - self.wall_start
        cpu_times = self.process.cpu_times()
        cpu_elapsed = (cpu_times.user + cpu_times.system) - self.cpu_start
        
        io_wait = max(0, wall_elapsed - cpu_elapsed)
        io_wait_pct = (io_wait / wall_elapsed * 100) if wall_elapsed > 0 else 0
        cpu_pct = (cpu_elapsed / wall_elapsed * 100) if wall_elapsed > 0 else 0
        
        return {
            'wall_time': wall_elapsed,
            'cpu_time': cpu_elapsed,
            'io_wait_time': io_wait,
            'io_wait_pct': io_wait_pct,
            'cpu_busy_pct': min(100, cpu_pct),
            'user_time': cpu_times.user - self.cpu_start,
            'system_time': cpu_times.system
        }


class DiskIOQueueMonitor:
    """
    Monitor disk I/O queue statistics.
    
    Tracks:
    - Read time: Total time spent on read operations (includes queue wait)
    - Read count: Number of read operations
    - Average read latency: Average time per read operation
    - Queue wait time estimate: Based on read time vs actual data transfer time
    """
    
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.samples: list = []
        self.disk_start = None
        self.disk_end = None
        self.start_time = 0.0
        
    def start(self):
        self.running = True
        self.samples = []
        self.start_time = time.time()
        
        # Get initial disk I/O counters
        try:
            self.disk_start = psutil.disk_io_counters()
        except Exception:
            self.disk_start = None
        
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        
        # Get final disk I/O counters
        try:
            self.disk_end = psutil.disk_io_counters()
        except Exception:
            self.disk_end = None
            
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        last_counters = self.disk_start
        last_time = time.time()
        
        while self.running:
            time.sleep(self.interval)
            
            try:
                current_counters = psutil.disk_io_counters()
                current_time = time.time()
                
                if last_counters and current_counters:
                    delta_time = current_time - last_time
                    
                    # Calculate delta values
                    delta_read_count = current_counters.read_count - last_counters.read_count
                    delta_read_bytes = current_counters.read_bytes - last_counters.read_bytes
                    delta_read_time = current_counters.read_time - last_counters.read_time  # in milliseconds
                    
                    # Calculate metrics
                    if delta_read_count > 0:
                        avg_read_latency_ms = delta_read_time / delta_read_count
                    else:
                        avg_read_latency_ms = 0
                    
                    # Estimate queue wait time
                    # Theoretical transfer time = bytes / (SSD speed ~3GB/s)
                    theoretical_transfer_time_ms = (delta_read_bytes / (3 * 1024 * 1024 * 1024)) * 1000
                    queue_wait_ms = max(0, delta_read_time - theoretical_transfer_time_ms)
                    
                    self.samples.append({
                        'timestamp': current_time,
                        'read_count': delta_read_count,
                        'read_bytes': delta_read_bytes,
                        'read_time_ms': delta_read_time,
                        'avg_latency_ms': avg_read_latency_ms,
                        'queue_wait_ms': queue_wait_ms,
                        'iops': delta_read_count / delta_time if delta_time > 0 else 0
                    })
                
                last_counters = current_counters
                last_time = current_time
                
            except Exception:
                pass
    
    def get_summary(self) -> dict:
        if not self.disk_start or not self.disk_end:
            return {
                'total_read_ops': 0,
                'total_read_time_ms': 0,
                'avg_read_latency_ms': 0,
                'queue_wait_time_ms': 0,
                'queue_wait_pct': 0,
                'avg_iops': 0
            }
        
        # Calculate totals from start to end
        total_read_ops = self.disk_end.read_count - self.disk_start.read_count
        total_read_bytes = self.disk_end.read_bytes - self.disk_start.read_bytes
        total_read_time_ms = self.disk_end.read_time - self.disk_start.read_time
        elapsed_time = time.time() - self.start_time
        
        # Average latency per operation
        avg_latency_ms = total_read_time_ms / total_read_ops if total_read_ops > 0 else 0
        
        # Estimate queue wait time
        # Theoretical transfer time at ~3GB/s SSD speed
        theoretical_transfer_ms = (total_read_bytes / (3 * 1024 * 1024 * 1024)) * 1000
        queue_wait_ms = max(0, total_read_time_ms - theoretical_transfer_ms)
        queue_wait_pct = (queue_wait_ms / total_read_time_ms * 100) if total_read_time_ms > 0 else 0
        
        # IOPS
        avg_iops = total_read_ops / elapsed_time if elapsed_time > 0 else 0
        
        # Get sample-based averages
        if self.samples:
            sample_avg_latency = sum(s['avg_latency_ms'] for s in self.samples) / len(self.samples)
            sample_avg_queue_wait = sum(s['queue_wait_ms'] for s in self.samples) / len(self.samples)
            peak_latency = max(s['avg_latency_ms'] for s in self.samples)
            peak_iops = max(s['iops'] for s in self.samples)
        else:
            sample_avg_latency = avg_latency_ms
            sample_avg_queue_wait = queue_wait_ms
            peak_latency = avg_latency_ms
            peak_iops = avg_iops
        
        return {
            'total_read_ops': total_read_ops,
            'total_read_time_ms': total_read_time_ms,
            'avg_read_latency_ms': avg_latency_ms,
            'sample_avg_latency_ms': sample_avg_latency,
            'peak_latency_ms': peak_latency,
            'queue_wait_time_ms': queue_wait_ms,
            'sample_avg_queue_wait_ms': sample_avg_queue_wait,
            'queue_wait_pct': queue_wait_pct,
            'avg_iops': avg_iops,
            'peak_iops': peak_iops,
            'total_read_bytes': total_read_bytes
        }


class CPUMonitor:
    """Monitor CPU utilization per core with on-CPU/off-CPU time tracking."""
    
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cpu_samples: list = []
        self.cpu_times_start: list = []
        self.cpu_times_end: list = []
        self.active_cores_threshold = 5.0
        self.start_time = 0.0
        self.end_time = 0.0
        
    def start(self):
        self.running = True
        self.cpu_samples = []
        self.start_time = time.time()
        
        # Capture initial per-CPU times
        self.cpu_times_start = psutil.cpu_times(percpu=True)
        
        # Initialize per-cpu measurement
        psutil.cpu_percent(percpu=True)
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        self.end_time = time.time()
        
        # Capture final per-CPU times
        self.cpu_times_end = psutil.cpu_times(percpu=True)
        
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        while self.running:
            cpu_percents = psutil.cpu_percent(percpu=True, interval=self.interval)
            active_cores = [i for i, pct in enumerate(cpu_percents) if pct > self.active_cores_threshold]
            
            self.cpu_samples.append({
                'timestamp': time.time(),
                'per_core': cpu_percents,
                'average': sum(cpu_percents) / len(cpu_percents),
                'active_cores': active_cores
            })
    
    def get_per_core_times(self) -> List[Dict]:
        """Calculate on-CPU and off-CPU time for each core."""
        if not self.cpu_times_start or not self.cpu_times_end:
            return []
        
        wall_time = self.end_time - self.start_time
        per_core_times = []
        
        for i, (start, end) in enumerate(zip(self.cpu_times_start, self.cpu_times_end)):
            # Calculate CPU time (user + system + nice)
            on_cpu_start = start.user + start.system + getattr(start, 'nice', 0)
            on_cpu_end = end.user + end.system + getattr(end, 'nice', 0)
            on_cpu_time = on_cpu_end - on_cpu_start
            
            # Calculate idle time
            idle_start = start.idle
            idle_end = end.idle
            idle_time = idle_end - idle_start
            
            # Off-CPU time is approximated from idle
            # Note: This is system-wide for this core, not just our process
            total_time = on_cpu_time + idle_time
            on_cpu_pct = (on_cpu_time / total_time * 100) if total_time > 0 else 0
            
            per_core_times.append({
                'core_id': i,
                'on_cpu_time': on_cpu_time,
                'idle_time': idle_time,
                'on_cpu_pct': on_cpu_pct,
                'off_cpu_pct': 100 - on_cpu_pct
            })
        
        return per_core_times
    
    def get_summary(self) -> dict:
        if not self.cpu_samples:
            return {
                'per_core_avg': [], 'overall_avg': 0.0, 
                'active_cores': [], 'cores_used': 0,
                'per_core_times': []
            }
        
        num_cores = len(self.cpu_samples[0]['per_core'])
        per_core_totals = [0.0] * num_cores
        all_active_cores = set()
        
        for sample in self.cpu_samples:
            for i, val in enumerate(sample['per_core']):
                per_core_totals[i] += val
            all_active_cores.update(sample['active_cores'])
        
        num_samples = len(self.cpu_samples)
        per_core_avg = [total / num_samples for total in per_core_totals]
        overall_avg = sum(per_core_avg) / len(per_core_avg)
        
        consistently_active = [i for i, avg in enumerate(per_core_avg) if avg > 10.0]
        
        return {
            'per_core_avg': per_core_avg,
            'overall_avg': overall_avg,
            'num_samples': num_samples,
            'active_cores': sorted(all_active_cores),
            'consistently_active_cores': consistently_active,
            'cores_used': len(consistently_active),
            'total_cores': num_cores,
            'per_core_times': self.get_per_core_times()
        }


class ThroughputMonitor:
    """Monitor and display disk throughput in real-time."""
    
    def __init__(self, stats: Stats, interval: float = 1.0):
        self.stats = stats
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.samples: list = []
    
    def start(self):
        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        last_bytes = 0
        last_time = time.time()
        
        while self.running:
            time.sleep(self.interval)
            current_bytes = self.stats.bytes_read
            current_time = time.time()
            
            delta_bytes = current_bytes - last_bytes
            delta_time = current_time - last_time
            
            if delta_time > 0:
                throughput_mbps = (delta_bytes / (1024 * 1024)) / delta_time
                self.samples.append(throughput_mbps)
                
                total_read_gb = current_bytes / (1024 * 1024 * 1024)
                print(f"\r[Progress] Read: {total_read_gb:.2f} GB | "
                      f"Throughput: {throughput_mbps:.2f} MB/s | "
                      f"Avg: {self.stats.get_throughput_mbps():.2f} MB/s", 
                      end='', flush=True)
            
            last_bytes = current_bytes
            last_time = current_time
    
    def get_summary(self) -> dict:
        if not self.samples:
            return {'avg_throughput': 0.0, 'peak_throughput': 0.0, 'min_throughput': 0.0}
        return {
            'avg_throughput': sum(self.samples) / len(self.samples),
            'peak_throughput': max(self.samples),
            'min_throughput': min(self.samples)
        }


def find_max_single_thread(filepath: str, chunk_size: int = 64 * 1024 * 1024,
                           stats: Optional[Stats] = None) -> float:
    """Find maximum number in file using single thread."""
    max_val = float('-inf')
    double_size = 8
    
    chunk_size = (chunk_size // double_size) * double_size
    
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            if stats:
                stats.add_bytes(len(chunk))
            
            num_doubles = len(chunk) // double_size
            if num_doubles > 0:
                doubles = struct.unpack(f'{num_doubles}d', chunk[:num_doubles * double_size])
                chunk_max = max(doubles)
                if chunk_max > max_val:
                    max_val = chunk_max
    
    return max_val


def find_max_in_range(filepath: str, start: int, end: int, 
                      chunk_size: int, stats: Optional[Stats] = None) -> float:
    """Find maximum number in a specific range of the file."""
    max_val = float('-inf')
    double_size = 8
    
    with open(filepath, 'rb') as f:
        f.seek(start)
        remaining = end - start
        
        while remaining > 0:
            to_read = min(chunk_size, remaining)
            chunk = f.read(to_read)
            if not chunk:
                break
            
            if stats:
                stats.add_bytes(len(chunk))
            
            remaining -= len(chunk)
            
            num_doubles = len(chunk) // double_size
            if num_doubles > 0:
                doubles = struct.unpack(f'{num_doubles}d', chunk[:num_doubles * double_size])
                chunk_max = max(doubles)
                if chunk_max > max_val:
                    max_val = chunk_max
    
    return max_val


def find_max_multi_thread(filepath: str, num_threads: int, 
                          chunk_size: int = 64 * 1024 * 1024,
                          stats: Optional[Stats] = None) -> float:
    """Find maximum number in file using multiple threads."""
    file_size = os.path.getsize(filepath)
    double_size = 8
    
    segment_size = (file_size // num_threads // double_size) * double_size
    
    segments = []
    for i in range(num_threads):
        start = i * segment_size
        if i == num_threads - 1:
            end = file_size
        else:
            end = (i + 1) * segment_size
        segments.append((start, end))
    
    max_val = float('-inf')
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for start, end in segments:
            future = executor.submit(find_max_in_range, filepath, start, end, 
                                    chunk_size, stats)
            futures.append(future)
        
        for future in as_completed(futures):
            result = future.result()
            if result > max_val:
                max_val = result
    
    return max_val


def print_tabulated_results(results: dict):
    """Print results in a nice tabulated format."""
    print(f"\n{'='*80}")
    print(f"{'BENCHMARK RESULTS SUMMARY':^80}")
    print(f"{'='*80}")
    
    # Main metrics table
    print(f"\n{'METRIC':<40} {'VALUE':>35}")
    print(f"{'-'*40} {'-'*35}")
    
    file_size_gb = results['file_size'] / (1024**3)
    print(f"{'File Size':<40} {file_size_gb:>32.2f} GB")
    print(f"{'Threads Used':<40} {results['num_threads']:>35}")
    print(f"{'Maximum Value Found':<40} {results['max_value']:>35g}")
    print(f"{'Total Execution Time':<40} {results['elapsed_time']:>32.2f} sec")
    
    print(f"\n{'--- DISK THROUGHPUT ---':<40}")
    print(f"{'Average Throughput':<40} {results['throughput']['avg_throughput']:>30.2f} MB/s")
    print(f"{'Peak Throughput':<40} {results['throughput']['peak_throughput']:>30.2f} MB/s")
    print(f"{'Minimum Throughput':<40} {results['throughput']['min_throughput']:>30.2f} MB/s")
    
    print(f"\n{'--- I/O TIMING ---':<40}")
    io = results['io_stats']
    print(f"{'Wall Clock Time':<40} {io['wall_time']:>32.2f} sec")
    print(f"{'CPU Time (user+sys)':<40} {io['cpu_time']:>32.2f} sec")
    print(f"{'  User Time':<40} {io['user_time']:>32.2f} sec")
    print(f"{'  System Time':<40} {io['system_time']:>32.2f} sec")
    print(f"{'I/O Wait Time (estimated)':<40} {io['io_wait_time']:>32.2f} sec")
    print(f"{'I/O Wait Percentage':<40} {io['io_wait_pct']:>32.1f} %")
    print(f"{'CPU Busy Percentage':<40} {io['cpu_busy_pct']:>32.1f} %")
    
    # Disk I/O Queue Statistics
    if 'disk_io' in results and results['disk_io']['total_read_ops'] > 0:
        dio = results['disk_io']
        print(f"\n{'--- DISK I/O QUEUE STATISTICS ---':<40}")
        print(f"{'Total Read Operations':<40} {dio['total_read_ops']:>35,}")
        print(f"{'Total Disk Read Time':<40} {dio['total_read_time_ms']:>32.2f} ms")
        print(f"{'Average Read Latency':<40} {dio['avg_read_latency_ms']:>32.3f} ms")
        print(f"{'Peak Read Latency':<40} {dio['peak_latency_ms']:>32.3f} ms")
        print(f"{'Queue Wait Time (estimated)':<40} {dio['queue_wait_time_ms']:>32.2f} ms")
        print(f"{'Queue Wait Percentage':<40} {dio['queue_wait_pct']:>32.1f} %")
        print(f"{'Average IOPS':<40} {dio['avg_iops']:>32.0f}")
        print(f"{'Peak IOPS':<40} {dio['peak_iops']:>32.0f}")
    
    print(f"\n{'--- CPU CORES ---':<40}")
    cpu = results['cpu']
    print(f"{'Total Logical Cores':<40} {cpu['total_cores']:>35}")
    print(f"{'Cores Actively Used':<40} {cpu['cores_used']:>35}")
    print(f"{'Overall CPU Utilization':<40} {cpu['overall_avg']:>33.1f} %")
    
    # Per-core table
    if cpu['per_core_times']:
        print(f"\n{'--- PER-CORE CPU TIME (On-CPU vs Off-CPU) ---'}")
        print(f"{'Core':<8} {'On-CPU Time':>14} {'Idle Time':>14} {'On-CPU %':>12} {'Off-CPU %':>12} {'Util %':>10}")
        print(f"{'-'*8} {'-'*14} {'-'*14} {'-'*12} {'-'*12} {'-'*10}")
        
        for i, core_time in enumerate(cpu['per_core_times']):
            util_pct = cpu['per_core_avg'][i] if i < len(cpu['per_core_avg']) else 0
            active_marker = "*" if i in cpu['consistently_active_cores'] else " "
            print(f"Core {i:<2}{active_marker} {core_time['on_cpu_time']:>13.2f}s {core_time['idle_time']:>13.2f}s "
                  f"{core_time['on_cpu_pct']:>11.1f}% {core_time['off_cpu_pct']:>11.1f}% {util_pct:>9.1f}%")
        
        print(f"  (* = consistently active core for this process)")
    
    print(f"\n{'='*80}\n")


def save_results_csv(results: dict, csv_path: str):
    """Append results to a CSV file for comparison across runs."""
    file_exists = os.path.exists(csv_path)
    
    row = {
        'timestamp': datetime.now().isoformat(),
        'file_size_gb': results['file_size'] / (1024**3),
        'num_threads': results['num_threads'],
        'max_value': results['max_value'],
        'execution_time_sec': results['elapsed_time'],
        'avg_throughput_mbps': results['throughput']['avg_throughput'],
        'peak_throughput_mbps': results['throughput']['peak_throughput'],
        'min_throughput_mbps': results['throughput']['min_throughput'],
        'wall_time_sec': results['io_stats']['wall_time'],
        'cpu_time_sec': results['io_stats']['cpu_time'],
        'user_time_sec': results['io_stats']['user_time'],
        'system_time_sec': results['io_stats']['system_time'],
        'io_wait_time_sec': results['io_stats']['io_wait_time'],
        'io_wait_pct': results['io_stats']['io_wait_pct'],
        'cpu_busy_pct': results['io_stats']['cpu_busy_pct'],
        'total_cores': results['cpu']['total_cores'],
        'cores_used': results['cpu']['cores_used'],
        'overall_cpu_util_pct': results['cpu']['overall_avg'],
        'active_core_ids': ','.join(map(str, results['cpu']['consistently_active_cores']))
    }
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    return csv_path


def save_results_json(results: dict, json_path: str):
    """Save detailed results to a JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'file_size_bytes': results['file_size'],
        'file_size_gb': results['file_size'] / (1024**3),
        'num_threads': results['num_threads'],
        'max_value': results['max_value'],
        'execution_time_sec': results['elapsed_time'],
        'throughput': results['throughput'],
        'io_stats': results['io_stats'],
        'cpu': {
            'total_cores': results['cpu']['total_cores'],
            'cores_used': results['cpu']['cores_used'],
            'overall_avg': results['cpu']['overall_avg'],
            'per_core_avg': results['cpu']['per_core_avg'],
            'active_cores': results['cpu']['consistently_active_cores'],
            'per_core_times': results['cpu']['per_core_times']
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    return json_path


def run_benchmark(filepath: str, num_threads: int = 1, 
                  chunk_size_mb: int = 64) -> dict:
    """Run the max finder with full monitoring."""
    chunk_size = chunk_size_mb * 1024 * 1024
    file_size = os.path.getsize(filepath)
    
    print(f"\n{'='*60}")
    print(f"Max Finder Benchmark (Python)")
    print(f"{'='*60}")
    print(f"File: {filepath}")
    print(f"File Size: {file_size / (1024**3):.2f} GB")
    print(f"Mode: {'Single-threaded' if num_threads == 1 else f'Multi-threaded ({num_threads} threads)'}")
    print(f"Chunk Size: {chunk_size_mb} MB")
    print(f"{'='*60}\n")
    
    # Initialize monitoring
    stats = Stats(start_time=time.time())
    cpu_monitor = CPUMonitor(interval=0.5)
    throughput_monitor = ThroughputMonitor(stats, interval=1.0)
    io_wait_monitor = IOWaitMonitor()
    disk_io_monitor = DiskIOQueueMonitor(interval=0.5)
    
    # Start monitors
    cpu_monitor.start()
    throughput_monitor.start()
    io_wait_monitor.start()
    disk_io_monitor.start()
    
    start_time = time.time()
    
    try:
        if num_threads == 1:
            max_val = find_max_single_thread(filepath, chunk_size, stats)
        else:
            max_val = find_max_multi_thread(filepath, num_threads, chunk_size, stats)
    finally:
        throughput_monitor.stop()
        cpu_monitor.stop()
        disk_io_monitor.stop()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Get summaries
    cpu_summary = cpu_monitor.get_summary()
    throughput_summary = throughput_monitor.get_summary()
    io_stats = io_wait_monitor.get_stats()
    disk_io_stats = disk_io_monitor.get_summary()
    
    results = {
        'max_value': max_val,
        'elapsed_time': elapsed,
        'throughput': throughput_summary,
        'cpu': cpu_summary,
        'io_stats': io_stats,
        'disk_io': disk_io_stats,
        'file_size': file_size,
        'num_threads': num_threads
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Find the maximum number in a large binary file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-threaded mode
  python max_finder.py data.bin --threads 1
  
  # Multi-threaded with 10 cores
  python max_finder.py data.bin --threads 10
  
  # All cores with CSV output
  python max_finder.py data.bin --threads 0 --csv results.csv
  
  # With JSON output for detailed analysis
  python max_finder.py data.bin --threads 4 --json results.json
        """
    )
    
    parser.add_argument('filepath', help='Path to the binary data file')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of threads (1=single-thread, 0=all cores, default: 1)')
    parser.add_argument('-c', '--chunk-size', type=int, default=64,
                        help='Chunk size in MB (default: 64)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Append results to CSV file for comparison')
    parser.add_argument('--json', type=str, default=None,
                        help='Save detailed results to JSON file')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filepath):
        print(f"Error: File '{args.filepath}' not found", file=sys.stderr)
        sys.exit(1)
    
    num_threads = args.threads
    if num_threads == 0:
        num_threads = psutil.cpu_count(logical=True)
        print(f"Using all {num_threads} logical cores")
    elif num_threads < 0:
        print(f"Error: Invalid thread count: {num_threads}", file=sys.stderr)
        sys.exit(1)
    
    try:
        results = run_benchmark(args.filepath, num_threads, args.chunk_size)
        
        # Print tabulated results
        print_tabulated_results(results)
        
        # Save to CSV if requested
        if args.csv:
            csv_path = save_results_csv(results, args.csv)
            print(f"Results appended to: {csv_path}")
        
        # Save to JSON if requested
        if args.json:
            json_path = save_results_json(results, args.json)
            print(f"Detailed results saved to: {json_path}")
        
        # Final summary line
        print(f"Summary: Found max={results['max_value']} in {results['elapsed_time']:.2f}s "
              f"@ {results['throughput']['avg_throughput']:.0f} MB/s avg throughput")
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
