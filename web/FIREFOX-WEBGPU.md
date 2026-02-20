# Firefox WebGPU Performance

Firefox's WebGPU implementation is significantly slower than Chrome and Safari
for GPU compute workloads. This affects all WebGPU-based inference, not just
this project.

## Observed Impact

**STT (this project), same machine:**

| Browser | RTF    | ms/frame | Mimi (CPU) | STT (GPU) |
|---------|--------|----------|------------|-----------|
| Chrome  | 1.00x  | ~80ms    | 27.8ms     | 12.7ms    |
| Firefox | 1.30x  | ~104ms   | 32.1ms     | 15.4ms    |

**WebLLM (independent WebGPU LLM benchmark):**

| Browser | Tokens/s | Relative |
|---------|----------|----------|
| Chrome  | 30.8     | 1.0x     |
| Safari  | 29.2     | 0.95x    |
| Firefox | 8.4      | 0.27x    |

Safari is on par with Chrome. The issue is Firefox-specific.

## Root Cause

Firefox Bug [1870699](https://bugzilla.mozilla.org/show_bug.cgi?id=1870699):
Firefox polls for GPU task completion on a **100ms timer** instead of using
event-driven notification. Every `mapAsync` readback pays up to 100ms of
latency waiting for the next poll tick, regardless of actual GPU execution time.

Additional contributing factors:
- wgpu Metal backend creates separate `MTLCommandBuffer` per encoder
  ([wgpu #6001](https://github.com/gfx-rs/wgpu/issues/6001))
- Unnecessary `Mutex` on Metal `CommandQueue`
  ([wgpu #5494](https://github.com/gfx-rs/wgpu/issues/5494))
- IPC command buffering was only added in Firefox 142
  ([Bug 1968122](https://bugzilla.mozilla.org/show_bug.cgi?id=1968122))

## What We Investigated

1. Confirmed the gap is ~23ms/frame: 7ms slower compute + 16ms poll overhead
2. cubecl auto-flushes every 32 dispatches (`tasks_max=32`), so most GPU work
   IS submitted during `submit_frame()` — pipelining partially masks the poll
3. An explicit flush after `submit_frame()` would gain ~5-10ms at best
4. Eliminating readback entirely (GPU-resident argmax + embedding lookup) could
   remove the poll overhead but requires significant refactoring
5. None of these workarounds address the fundamental 3.7x slowdown seen in
   independent benchmarks

## Conclusion

The fix is in Mozilla's hands. Monitor Bug 1870699 for event-driven GPU polling.
Test periodically with Firefox Nightly for improvements.
