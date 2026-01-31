# Large File Maximum Number Finder

A Python benchmarking tool for finding the maximum number in large binary files, optimized for Apple Silicon Macs (M4 chip).

## Features

- **Single-threaded mode**: Sequential scan using one core
- **Multi-threaded mode**: Parallel scan using configurable number of cores
- **Real-time monitoring**: Disk throughput and per-core CPU utilization
- **Comprehensive benchmarking**: Compare different configurations

## Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Test Data

Multiple generators are available with different speed/randomness trade-offs:

```bash
# FASTEST: Pattern mode (~500-800 MB/s) - best for I/O benchmarking
python generate_fastest.py data_200gb.bin --size 200 --mode pattern --known-max 1e100

# FAST: Parallel random (~400-600 MB/s)
python generate_fastest.py data_200gb.bin --size 200 --mode random --known-max 1e100

# STANDARD: Single-threaded random (~150-250 MB/s)
python generate_data.py data_200gb.bin --size 200 --known-max 1e100
```

**Generator Comparison:**

| Generator | Speed | Data Type | Best For |
|-----------|-------|-----------|----------|
| `generate_fastest.py --mode pattern` | 500-800 MB/s | Repeating pattern | Pure I/O benchmarks |
| `generate_fastest.py --mode random` | 400-600 MB/s | Full random | General testing |
| `generate_data_ultra.py` | 400-500 MB/s | Full random (mmap) | Large files |
| `generate_data.py` | 150-250 MB/s | Full random | Simple, reliable |

**Estimated time for 200GB:**
- Pattern mode: ~4-7 minutes
- Random mode: ~6-10 minutes
- Standard mode: ~15-25 minutes

### 2. Run Max Finder

```bash
# Single-threaded mode
python max_finder.py test_1gb.bin --threads 1

# Multi-threaded with 4 cores
python max_finder.py test_1gb.bin --threads 4

# Multi-threaded with 8 cores
python max_finder.py test_1gb.bin --threads 8

# Use all available cores
python max_finder.py test_1gb.bin --threads 0
```

### 3. Run Comprehensive Benchmark

```bash
# Compare 1, 2, 4, and 8 threads
python benchmark.py test_1gb.bin --threads 1 2 4 8

# Include all cores and save results
python benchmark.py test_1gb.bin --threads 1 2 4 8 --all-cores --output results.json

# Multiple runs for statistical significance
python benchmark.py test_1gb.bin --threads 1 4 8 --runs 3
```

## Command Line Options

### max_finder.py

| Option | Description |
|--------|-------------|
| `filepath` | Path to binary data file (required) |
| `-t, --threads N` | Number of threads (1=single, 0=all cores) |
| `-c, --chunk-size N` | Chunk size in MB (default: 64) |
| `-q, --quiet` | Suppress progress output |

### generate_fastest.py (Recommended)

| Option | Description |
|--------|-------------|
| `filepath` | Output file path (required) |
| `-s, --size N` | File size in GB (required) |
| `-m, --mode` | Generation mode: `random`, `fast-random`, `pattern` |
| `-w, --workers N` | Number of parallel workers (default: all CPUs) |
| `--known-max N` | Insert known max for verification |
| `--max-position` | Position for known max (start/middle/end/random) |

### generate_data.py (Simple)

| Option | Description |
|--------|-------------|
| `filepath` | Output file path (required) |
| `-s, --size N` | File size in GB (required) |
| `-c, --chunk-size N` | Write chunk size in MB (default: 256) |
| `--seed N` | Random seed for reproducibility |
| `--known-max N` | Insert known max for verification |
| `--max-position` | Position for known max (start/middle/end/random) |

### benchmark.py

| Option | Description |
|--------|-------------|
| `filepath` | Test data file (required) |
| `-t, --threads` | Thread counts to test (e.g., 1 2 4 8) |
| `-c, --chunk-sizes` | Chunk sizes in MB to test |
| `-r, --runs N` | Runs per configuration |
| `-o, --output` | JSON file for results |
| `--all-cores` | Include test with all cores |

## Output Metrics

### Disk Throughput
- **Average**: Mean throughput during scan
- **Peak**: Maximum observed throughput
- **Overall**: Total bytes / total time

### CPU Utilization
- **Per-core averages**: Shows utilization for each CPU core
- **Overall average**: Mean across all cores

## File Format

The data files use binary format with double-precision floating point numbers (8 bytes each):
- Each number is stored as IEEE 754 double (64-bit)
- Numbers are stored sequentially with no delimiters
- File size = number_of_values × 8 bytes

## Performance Tips for M4 Mac

1. **Chunk Size**: Start with 64 MB, try 32-128 MB for your SSD
2. **Thread Count**: M4 has 10 cores (4 performance + 6 efficiency), try 4-10 threads
3. **SSD Performance**: Modern Mac SSDs can achieve 3-7 GB/s sequential read
4. **Memory**: Ensure sufficient RAM to avoid swapping

## Example Output

```
============================================================
Max Finder Benchmark
============================================================
File: test_10gb.bin
File Size: 10.00 GB
Mode: Multi-threaded (8 threads)
Chunk Size: 64 MB
============================================================

[Progress] Read: 10.00 GB | Throughput: 3542.15 MB/s | Avg: 3498.22 MB/s

============================================================
RESULTS
============================================================
Maximum Value Found: 1e+100
Total Time: 2.92 seconds

--- Disk Throughput ---
Average: 3498.22 MB/s
Peak: 3891.45 MB/s
Minimum: 2987.11 MB/s
Overall: 3498.22 MB/s

--- CPU Utilization ---
Overall Average: 42.3%
Per-Core Averages:
  Core  0: 78.2%  |  Core  1: 75.4%  |  Core  2: 72.1%  |  Core  3: 71.8%
  Core  4: 35.2%  |  Core  5: 33.1%  |  Core  6: 31.8%  |  Core  7: 28.4%
  Core  8: 18.2%  |  Core  9: 15.1%
============================================================
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Main Process                         │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ CPU Monitor  │  │  Throughput  │  │   Worker     │  │
│  │   Thread     │  │   Monitor    │  │   Threads    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                 │                 │          │
│         ▼                 ▼                 ▼          │
│  ┌────────────────────────────────────────────────┐    │
│  │              Shared Statistics                 │    │
│  │         (thread-safe with locks)               │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   200 GB Data File                      │
│  ┌─────────┬─────────┬─────────┬─────────┬──────────┐  │
│  │Segment 1│Segment 2│Segment 3│   ...   │Segment N │  │
│  │(Thread1)│(Thread2)│(Thread3)│         │(ThreadN) │  │
│  └─────────┴─────────┴─────────┴─────────┴──────────┘  │
└─────────────────────────────────────────────────────────┘
```

## License

MIT License - see LICENSE file for details.
