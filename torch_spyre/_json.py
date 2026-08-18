# Copyright 2025-2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import functools
import json as _std_json
import threading
import time
from typing import Any, Callable


@dataclasses.dataclass
class _Stat:
    calls: int = 0
    total_time_seconds: float = 0.0


_lock = threading.Lock()
_stats: dict[str, _Stat] = {
    "load": _Stat(),
    "loads": _Stat(),
    "dump": _Stat(),
    "dumps": _Stat(),
}


def _instrument(name: str, fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            with _lock:
                stat = _stats[name]
                stat.calls += 1
                stat.total_time_seconds += elapsed

    return wrapper


load = _instrument("load", _std_json.load)
loads = _instrument("loads", _std_json.loads)
dump = _instrument("dump", _std_json.dump)
dumps = _instrument("dumps", _std_json.dumps)

# Expose remaining attributes from standard json module
JSONDecodeError = _std_json.JSONDecodeError
JSONDecoder = _std_json.JSONDecoder
JSONEncoder = _std_json.JSONEncoder


def print_json_statistics() -> None:
    """Print the global statistics for json load, loads, dump, and dumps calls."""
    with _lock:
        snapshot = {name: (stat.calls, stat.total_time_seconds) for name, stat in _stats.items()}

    print("=" * 60)
    print(f"{'JSON Function':<15} {'Calls':<12} {'Total Time (s)':<18} {'Avg Time (ms)':<15}")
    print("-" * 60)
    total_calls = 0
    total_time = 0.0
    for name, (calls, time_sec) in snapshot.items():
        total_calls += calls
        total_time += time_sec
        avg_ms = (time_sec / calls * 1000.0) if calls > 0 else 0.0
        print(f"{name:<15} {calls:<12} {time_sec:<18.6f} {avg_ms:<15.4f}")
    print("-" * 60)
    total_avg_ms = (total_time / total_calls * 1000.0) if total_calls > 0 else 0.0
    print(f"{'Total':<15} {total_calls:<12} {total_time:<18.6f} {total_avg_ms:<15.4f}")
    print("=" * 60)


def reset_json_statistics() -> None:
    """Reset all json metrics counters."""
    with _lock:
        for stat in _stats.values():
            stat.calls = 0
            stat.total_time_seconds = 0.0
