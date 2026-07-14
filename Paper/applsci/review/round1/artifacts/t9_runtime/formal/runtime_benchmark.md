# T9 Runtime Benchmark

Architecture-level inference benchmark on synthetic inputs. No checkpoint weights were loaded.
Parameter scope note: `Params(reg)` counts only the registered restoration module. `Params(full)` adds the RAFT flow estimator used by the executable forward pass.
The 1280x720 batch-4 OOM rows were observed in a shared GPU session with substantial pre-existing memory usage, so they are reported as session observations rather than clean-device capacity bounds.

- Timestamp: `2026-06-05T18:28:10.738222+00:00`
- Device: `cuda`
- PyTorch: `2.7.0+cu126`
- Protocol: FP32, eval mode, inference_mode, warmup=3, runs=10

| Model | Batch | Resolution | Status | Latency ms/clip | ms/frame | Clips/s | Frames/s | Peak allocated MB | Params(reg) M | Params(full) M |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| RTN baseline | 1 | 256x256 | ok | 188.740 | 26.963 | 5.298 | 37.088 | 327.7 | 0.9212 | 6.1783 |
| RTN baseline | 4 | 256x256 | ok | 129.935 | 18.562 | 7.696 | 53.873 | 1180.0 | 0.9212 | 6.1783 |
| RTN baseline | 1 | 1280x720 | ok | 4174.462 | 596.352 | 0.240 | 1.677 | 10162.9 | 0.9212 | 6.1783 |
| RTN baseline | 4 | 1280x720 | oom: CUDA out of memory. Tried to allocate 18.54 GiB. GPU 0 has a total capacity of 44.40 GiB of which 3.66 GiB is free. Including non-PyTorch memory, this process has 40.73 GiB memory in use. Of the allocated memory 21.49 GiB is allocated by PyTorch, and 18.75 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) |  |  |  |  |  | 0.9212 | 6.1783 |
| WaveletMask | 1 | 256x256 | ok | 183.299 | 26.186 | 5.456 | 38.189 | 329.0 | 0.9221 | 6.1792 |
| WaveletMask | 4 | 256x256 | ok | 131.856 | 18.837 | 7.584 | 53.088 | 1180.1 | 0.9221 | 6.1792 |
| WaveletMask | 1 | 1280x720 | ok | 4171.793 | 595.970 | 0.240 | 1.678 | 10162.2 | 0.9221 | 6.1792 |
| WaveletMask | 4 | 1280x720 | oom: CUDA out of memory. Tried to allocate 18.54 GiB. GPU 0 has a total capacity of 44.40 GiB of which 2.01 GiB is free. Including non-PyTorch memory, this process has 42.38 GiB memory in use. Of the allocated memory 21.49 GiB is allocated by PyTorch, and 20.40 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) |  |  |  |  |  | 0.9221 | 6.1792 |
