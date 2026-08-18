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
