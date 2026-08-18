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
import threading
import time
from typing import Any, Callable

import orjson


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


def _loads_adapter(s: str | bytes, *args: Any, **kwargs: Any) -> Any:
    return orjson.loads(s)


def _dumps_adapter(
    obj: Any,
    *args: Any,
    indent: int | None = None,
    default: Callable[[Any], Any] | None = None,
    sort_keys: bool = False,
    **kwargs: Any,
) -> str:
    option = 0
    if indent:
        option |= orjson.OPT_INDENT_2
    if sort_keys:
        option |= orjson.OPT_SORT_KEYS

    b = orjson.dumps(obj, default=default, option=option if option else None)
    return b.decode("utf-8")


def _dump_adapter(
    obj: Any,
    fp: Any,
    *args: Any,
    indent: int | None = None,
    default: Callable[[Any], Any] | None = None,
    sort_keys: bool = False,
    **kwargs: Any,
) -> None:
    s = _dumps_adapter(obj, *args, indent=indent, default=default, sort_keys=sort_keys, **kwargs)
    fp.write(s)


def _load_adapter(fp: Any, *args: Any, **kwargs: Any) -> Any:
    content = fp.read()
    return _loads_adapter(content)


load = _instrument("load", _load_adapter)
loads = _instrument("loads", _loads_adapter)
dump = _instrument("dump", _dump_adapter)
dumps = _instrument("dumps", _dumps_adapter)


def get_json_statistics() -> dict[str, dict[str, Any]]:
    """Return a dictionary containing both Python frontend and C++ backend JSON metrics."""
    with _lock:
        py_snapshot = {
            name: {"calls": stat.calls, "total_time_seconds": stat.total_time_seconds}
            for name, stat in _stats.items()
        }

    cpp_stats = {}
    try:
        from torch_spyre import _C

        if hasattr(_C, "_get_cpp_json_statistics"):
            cpp_stats = _C._get_cpp_json_statistics()
    except Exception:
        pass

    return {
        "python": py_snapshot,
        "cpp": cpp_stats,
    }


def print_json_statistics() -> None:
    """Print the global statistics for Python and C++ JSON operations."""
    stats = get_json_statistics()
    py_snapshot = stats["python"]
    cpp_snapshot = stats["cpp"]

    print("=" * 65)
    print(f"{'Python JSON Function':<22} {'Calls':<10} {'Total Time (s)':<18} {'Avg Time (ms)':<15}")
    print("-" * 65)
    py_total_calls = 0
    py_total_time = 0.0
    for name, data in py_snapshot.items():
        calls = data["calls"]
        time_sec = data["total_time_seconds"]
        py_total_calls += calls
        py_total_time += time_sec
        avg_ms = (time_sec / calls * 1000.0) if calls > 0 else 0.0
        print(f"{name:<22} {calls:<10} {time_sec:<18.6f} {avg_ms:<15.4f}")
    print("-" * 65)
    py_avg_ms = (py_total_time / py_total_calls * 1000.0) if py_total_calls > 0 else 0.0
    print(f"{'Python Total':<22} {py_total_calls:<10} {py_total_time:<18.6f} {py_avg_ms:<15.4f}")

    if cpp_snapshot:
        print("=" * 65)
        print(f"{'C++ JSON Operation':<22} {'Calls':<10} {'Total Time (s)':<18} {'Avg Time (ms)':<15}")
        print("-" * 65)
        parse_calls = cpp_snapshot.get("parse_calls", 0)
        parse_time = cpp_snapshot.get("parse_time_seconds", 0.0)
        parse_avg_ms = (parse_time / parse_calls * 1000.0) if parse_calls > 0 else 0.0
        print(f"{'spyrecode.json parse':<22} {parse_calls:<10} {parse_time:<18.6f} {parse_avg_ms:<15.4f}")

        read_calls = cpp_snapshot.get("file_read_calls", 0)
        read_time = cpp_snapshot.get("file_read_time_seconds", 0.0)
        read_avg_ms = (read_time / read_calls * 1000.0) if read_calls > 0 else 0.0
        print(f"{'spyrecode.json read':<22} {read_calls:<10} {read_time:<18.6f} {read_avg_ms:<15.4f}")

        cpp_total_calls = parse_calls + read_calls
        cpp_total_time = parse_time + read_time
        cpp_avg_ms = (cpp_total_time / cpp_total_calls * 1000.0) if cpp_total_calls > 0 else 0.0
        print("-" * 65)
        print(f"{'C++ Total':<22} {cpp_total_calls:<10} {cpp_total_time:<18.6f} {cpp_avg_ms:<15.4f}")

    print("=" * 65)


def reset_json_statistics() -> None:
    """Reset all Python and C++ json metrics counters."""
    with _lock:
        for stat in _stats.values():
            stat.calls = 0
            stat.total_time_seconds = 0.0
    try:
        from torch_spyre import _C

        if hasattr(_C, "_reset_cpp_json_statistics"):
            _C._reset_cpp_json_statistics()
    except Exception:
        pass
