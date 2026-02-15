"""
GPU-accelerated SCD and FFT computation using OpenCL.
Works with AMD, NVIDIA, and Intel GPUs that support OpenCL.

For AMD Radeon GPUs on Windows: install pyopencl via
    pip install pyopencl

AMD GPUs ship with OpenCL drivers by default on Windows.
"""

import numpy as np

_GPU_AVAILABLE = False
_cl = None

try:
    import pyopencl as cl
    _cl = cl
    _GPU_AVAILABLE = True
except ImportError:
    pass


# ──────────────────── OpenCL Kernel Source ────────────────────
_KERNEL_SRC = r"""
// ─── SCD batch cross-correlation kernel ───
// Each work item computes one (alpha, freq) bin of the SCD.
// Supports both even m (exact half-bin) and odd m (interpolated).
// Outputs magnitude directly to avoid a second pass.
__kernel void scd_correlate_batch(
    __global const float2 *ffts,     // [S, N] complex FFTs, row-major
    __global       float  *scd_mag,  // [num_alphas, N] magnitude output
    __global const int    *m_vals,   // [num_alphas] alpha bin offsets
    const int N,
    const int S,
    const int num_alphas
) {
    const int k         = get_global_id(0);   // frequency bin
    const int alpha_idx = get_global_id(1);   // alpha index

    if (alpha_idx >= num_alphas || k >= N) return;

    const int m = m_vals[alpha_idx];
    const int is_odd = (m & 1) != 0;

    float sum_re = 0.0f;
    float sum_im = 0.0f;

    if (!is_odd) {
        // ── Even m: exact integer half-offset ──
        const int mh     = m / 2;
        const int abs_mh = mh < 0 ? -mh : mh;
        const int k0     = abs_mh;
        const int k1     = N - abs_mh;

        if (k < k0 || k >= k1) {
            scd_mag[alpha_idx * N + k] = 0.0f;
            return;
        }

        for (int s = 0; s < S; ++s) {
            const int base = s * N;
            const float2 xp = ffts[base + k + mh];
            const float2 xm = ffts[base + k - mh];
            sum_re += xp.x * xm.x + xp.y * xm.y;
            sum_im += xp.y * xm.x - xp.x * xm.y;
        }
    } else {
        // ── Odd m: linear interpolation ──
        // m/2.0 falls between floor and ceil -> average adjacent bins
        const int mh_floor = m / 2;
        const int mh_ceil  = (m > 0) ? mh_floor + 1 : mh_floor - 1;
        const int abs_f = mh_floor < 0 ? -mh_floor : mh_floor;
        const int abs_c = mh_ceil  < 0 ? -mh_ceil  : mh_ceil;
        const int abs_max  = abs_f > abs_c ? abs_f : abs_c;
        const int k0       = abs_max;
        const int k1       = N - abs_max;

        if (k < k0 || k >= k1) {
            scd_mag[alpha_idx * N + k] = 0.0f;
            return;
        }

        for (int s = 0; s < S; ++s) {
            const int base = s * N;
            // Interpolated X(f + m/2)
            const float2 a = ffts[base + k + mh_floor];
            const float2 b = ffts[base + k + mh_ceil];
            const float2 xp = (float2)(0.5f*(a.x+b.x), 0.5f*(a.y+b.y));
            // Interpolated X(f - m/2)
            const float2 c = ffts[base + k - mh_floor];
            const float2 d = ffts[base + k - mh_ceil];
            const float2 xm = (float2)(0.5f*(c.x+d.x), 0.5f*(c.y+d.y));
            // xp * conj(xm)
            sum_re += xp.x * xm.x + xp.y * xm.y;
            sum_im += xp.y * xm.x - xp.x * xm.y;
        }
    }

    const float inv_S = 1.0f / (float)S;
    sum_re *= inv_S;
    sum_im *= inv_S;

    scd_mag[alpha_idx * N + k] = sqrt(sum_re * sum_re + sum_im * sum_im);
}

// ─── Batch FFT windowing (multiply signal segments by window) ───
// Each work item processes one (segment, sample) pair.
__kernel void apply_window_batch(
    __global const float2 *signal,   // contiguous complex samples
    __global const float  *window,   // [N] window function
    __global       float2 *out,      // [num_segs, N] windowed segments
    const int N,
    const int stride,                // N - overlap
    const int num_segs
) {
    const int seg = get_global_id(1);
    const int n   = get_global_id(0);

    if (seg >= num_segs || n >= N) return;

    const int src_idx = seg * stride + n;
    const float w = window[n];
    const float2 s = signal[src_idx];

    out[seg * N + n] = (float2)(s.x * w, s.y * w);
}
"""


class GPUCompute:
    """GPU-accelerated SCD computation via OpenCL (AMD / NVIDIA / Intel)."""

    def __init__(self):
        self.ctx = None
        self.queue = None
        self.program = None
        self.available = False
        self.device_name = "None"
        self.platform_name = "None"
        self._init_error = None

        if not _GPU_AVAILABLE:
            self._init_error = "pyopencl not installed"
            return

        try:
            platforms = _cl.get_platforms()
            for plat in platforms:
                try:
                    devices = plat.get_devices(device_type=_cl.device_type.GPU)
                except _cl.LogicError:
                    continue
                if devices:
                    self.ctx = _cl.Context(devices=[devices[0]])
                    self.queue = _cl.CommandQueue(self.ctx)
                    self.device_name = devices[0].name.strip()
                    self.platform_name = plat.name.strip()
                    self._build_kernels()
                    self.available = True
                    break
            if not self.available:
                self._init_error = "No OpenCL GPU device found"
        except Exception as e:
            self._init_error = str(e)
            self.available = False

    # ─────────────────────────────────────────
    def _build_kernels(self):
        self.program = _cl.Program(self.ctx, _KERNEL_SRC).build()
        # Cache kernel objects to avoid repeated retrieval warning
        self._kern_scd_batch = _cl.Kernel(self.program, "scd_correlate_batch")

    # ─────────────────────────────────────────
    def status_string(self):
        """Return a short human-readable status."""
        if self.available:
            return f"GPU: {self.device_name} ({self.platform_name})"
        return f"GPU unavailable: {self._init_error}"

    # ─────────────────────────────────────────
    def compute_scd_gpu(self, ffts_complex, m_vals, N):
        """
        GPU-accelerated SCD cross-correlation.

        Args
        ----
        ffts_complex : np.ndarray, shape [S, N], dtype complex64
            fftshift-ed FFTs of each segment.
        m_vals : np.ndarray, dtype int32
            Even-valued alpha bin offsets.
        N : int
            FFT size.

        Returns
        -------
        scd_mag : np.ndarray, shape [num_alphas, N], dtype float32
        """
        if not self.available:
            raise RuntimeError("GPU not available")

        S = ffts_complex.shape[0]
        num_alphas = len(m_vals)

        # Complex64 → interleaved float32 pairs (OpenCL float2)
        ffts_f2 = np.ascontiguousarray(
            ffts_complex.astype(np.complex64).view(np.float32)
        )
        m_vals_i32 = np.ascontiguousarray(m_vals.astype(np.int32))

        # Output buffer
        scd_mag = np.empty(num_alphas * N, dtype=np.float32)

        mf = _cl.mem_flags
        ffts_buf = _cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ffts_f2)
        m_buf    = _cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_vals_i32)
        scd_buf  = _cl.Buffer(self.ctx, mf.WRITE_ONLY, scd_mag.nbytes)

        global_size = (N, num_alphas)
        self._kern_scd_batch.set_args(
            ffts_buf, scd_buf, m_buf,
            np.int32(N), np.int32(S), np.int32(num_alphas),
        )
        _cl.enqueue_nd_range_kernel(self.queue, self._kern_scd_batch, global_size, None)

        _cl.enqueue_copy(self.queue, scd_mag, scd_buf)
        self.queue.finish()

        return scd_mag.reshape(num_alphas, N)
