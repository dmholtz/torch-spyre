/*
 * Copyright 2025 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Registers a ProfilerStubs implementation for the Spyre PrivateUse1 device.
//
// PyTorch's KINETO_PRIVATEUSE1_FALLBACK profiler state brackets every
// TorchOp with a pair of record()/elapsed() calls (one at op start, one at
// op end) to measure device-side time.  Without a registered stub, both
// fallback_start and fallback_end remain nullptr, privateuse1ElapsedUs()
// returns -1, no kernel duration is appended to the FunctionEvent, and the
// "SPYRE total" / "Self SPYRE" table columns are silently omitted.
//
// Spyre does not expose CUDA-style device events, so we use CPU wall-clock
// timestamps as a proxy.  Each ProfilerVoidEventStub carries a shared_ptr
// to an int64_t nanosecond timestamp; elapsed() returns the difference in
// microseconds.

#include <c10/util/ApproximateClock.h>
#include <torch/csrc/profiler/stubs/base.h>

#include "../spyre_stream.h"

namespace torch::profiler::impl {
namespace {

// A timestamp stored inside a ProfilerVoidEventStub (shared_ptr<void>).
// We allocate an int64_t on the heap and wrap it so the base class machinery
// (which only knows about shared_ptr<void>) can hold and release it.
struct SpyreTimestamp {
  int64_t ns{0};
};

struct SpyreProfilerStubs : public ProfilerStubs {
  void record(
      c10::DeviceIndex* device,
      ProfilerVoidEventStub* event,
      int64_t* cpu_ns) const override {
    int64_t now = c10::getTime();
    *event = std::make_shared<SpyreTimestamp>(SpyreTimestamp{now});
    if (device) {
      *device = 0;
    }
    if (cpu_ns) {
      *cpu_ns = now;
    }
  }

  float elapsed(
      const ProfilerVoidEventStub* start,
      const ProfilerVoidEventStub* end) const override {
    if (!start || !*start || !end || !*end) {
      return 0.0f;
    }
    auto* t0 = static_cast<SpyreTimestamp*>(start->get());
    auto* t1 = static_cast<SpyreTimestamp*>(end->get());
    // getTime() returns nanoseconds; elapsed() must return microseconds.
    return static_cast<float>(t1->ns - t0->ns) / 1000.0f;
  }

  void mark(const char*) const override {}
  void rangePush(const char*) const override {}
  void rangePop() const override {}

  bool enabled() const override {
    return true;
  }

  void onEachDevice(std::function<void(int)> op) const override {
    op(0);
  }

  void synchronize() const override {
    spyre::synchronizeDevice(std::nullopt);
  }
};

// Register at static-init time, before any profiling session can start.
struct RegisterSpyreProfilerStubs {
  RegisterSpyreProfilerStubs() {
    static SpyreProfilerStubs stubs;
    registerPrivateUse1Methods(&stubs);
  }
};
RegisterSpyreProfilerStubs reg;

}  // namespace
}  // namespace torch::profiler::impl
