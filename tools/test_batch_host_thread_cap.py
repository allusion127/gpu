#!/usr/bin/env python3
"""Static contract for separating CUDA arena width from CPU Driver workers."""
from pathlib import Path


source = (Path(__file__).resolve().parents[1] / "src" / "main.cpp").read_text(encoding="utf-8")
required = (
    'RASBERY_BATCH_HOST_THREADS',
    'rasberyVisibleCpuThreads()',
    'sched_getaffinity',
    'const int startup_visible_cpus = rasberyVisibleCpuThreads();',
    'const int visible_cpus = startup_visible_cpus;',
    'if (const char* host_env = std::getenv("RASBERY_BATCH_HOST_THREADS"))',
    'num_threads(host_threads)',
    '[RASBERY][BATCH_HOST]',
    '\\"arena_width\\"',
    '\\"host_threads\\"',
    '\\"visible_cpus\\"',
)
missing = [token for token in required if token not in source]
if missing:
    raise SystemExit(f"batch host thread cap: FAIL missing={missing}")
if "num_threads(batch_width)" in source:
    raise SystemExit("batch host thread cap: FAIL stale num_threads(batch_width)")
print("batch host thread cap: PASS")
