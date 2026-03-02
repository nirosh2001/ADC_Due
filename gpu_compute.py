"""
GPU-accelerated SCD and FFT computation using CuPy (NVIDIA CUDA) or OpenCL.
Prioritizes CuPy for NVIDIA GPUs (faster), falls back to OpenCL for AMD/Intel.

For NVIDIA GPUs: pip install cupy-cuda11x  (or cuda12x for newer drivers)
For AMD/Intel GPUs: pip install pyopencl
"""

import numpy as np

# Try CuPy first (NVIDIA CUDA - fastest)
_CUPY_AVAILABLE = False
_cp = None
try:
    import cupy as cp
    _cp = cp
    _CUPY_AVAILABLE = True
except ImportError:
    pass

# Fallback to OpenCL (AMD/NVIDIA/Intel - portable)
_OPENCL_AVAILABLE = False
_cl = None
try:
    import pyopencl as cl
    _cl = cl
    _OPENCL_AVAILABLE = True
except ImportError:
    pass

_GPU_AVAILABLE = _CUPY_AVAILABLE or _OPENCL_AVAILABLE


# ══════════════════════════════════════════════════════════════
# CuPy Implementation (NVIDIA CUDA - Fastest)
# ══════════════════════════════════════════════════════════════

class CuPyBackend:
    """GPU acceleration using CuPy (NVIDIA CUDA only)."""
    
    def __init__(self):
        self.available = False
        self.device_name = "None"
        self.platform_name = "CuPy/CUDA"
        self._init_error = None
        
        if not _CUPY_AVAILABLE:
            self._init_error = "CuPy not installed"
            return
            
        try:
            # Check CUDA availability
            device = _cp.cuda.Device(0)
            self.device_name = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
            self.available = True
        except Exception as e:
            self._init_error = f"CuPy init failed: {e}"
    
    def compute_scd_gpu(self, ffts_complex, m_vals, N):
        """GPU-accelerated SCD using CuPy.
        
        Args:
            ffts_complex: np.ndarray [S, N] complex64, fftshifted FFTs
            m_vals: np.ndarray int32, alpha bin offsets
            N: int, FFT size
            
        Returns:
            scd_mag: np.ndarray [num_alphas, N] float32
        """
        S = ffts_complex.shape[0]
        num_alphas = len(m_vals)
        
        # Transfer to GPU
        ffts_gpu = _cp.asarray(ffts_complex, dtype=_cp.complex64)
        m_vals_gpu = _cp.asarray(m_vals, dtype=_cp.int32)
        
        # Preallocate output
        scd_gpu = _cp.zeros((num_alphas, N), dtype=_cp.complex64)
        
        # Compute correlations for each alpha
        for ai in range(num_alphas):
            m = int(m_vals_gpu[ai])
            mh = m // 2
            abs_mh = abs(mh)
            k0 = abs_mh
            k1 = N - abs_mh
            
            if k1 > k0:
                # Vectorized cross-correlation on GPU
                Xp = ffts_gpu[:, k0 + mh:k1 + mh]
                Xm = ffts_gpu[:, k0 - mh:k1 - mh]
                scd_gpu[ai, k0:k1] = _cp.mean(Xp * _cp.conj(Xm), axis=0)
        
        # Magnitude and transfer back
        scd_mag = _cp.abs(scd_gpu).astype(_cp.float32)
        return _cp.asnumpy(scd_mag)
    
    def compute_fft_batch_gpu(self, segments, window):
        """Batch FFT computation on GPU.
        
        Args:
            segments: np.ndarray [S, N] complex64, signal segments
            window: np.ndarray [N] float32, window function
            
        Returns:
            ffts: np.ndarray [S, N] complex64, fftshifted FFTs
        """
        # Transfer to GPU
        segments_gpu = _cp.asarray(segments, dtype=_cp.complex64)
        window_gpu = _cp.asarray(window, dtype=_cp.float32)
        
        # Apply window (broadcast multiply)
        windowed = segments_gpu * window_gpu[None, :]
        
        # Batch FFT
        ffts_gpu = _cp.fft.fft(windowed, axis=1)
        
        # FFT shift
        ffts_shifted_gpu = _cp.fft.fftshift(ffts_gpu, axes=1)
        
        # Transfer back
        return _cp.asnumpy(ffts_shifted_gpu)


# ══════════════════════════════════════════════════════════════
# OpenCL Implementation (AMD/NVIDIA/Intel - Portable Fallback)
# ══════════════════════════════════════════════════════════════


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


class OpenCLBackend:
    """GPU-accelerated SCD computation via OpenCL (AMD / NVIDIA / Intel)."""

    def __init__(self):
        self.ctx = None
        self.queue = None
        self.program = None
        self.available = False
        self.device_name = "None"
        self.platform_name = "None"
        self._init_error = None

        if not _OPENCL_AVAILABLE:
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
    
    def compute_fft_batch_gpu(self, segments, window):
        """Batch FFT computation (OpenCL doesn't support FFT natively).
        
        This is a placeholder - OpenCL would need clFFT library.
        Falls back to CPU FFT for now.
        """
        # OpenCL FFT requires additional library (clFFT)
        # For now return None to signal CPU fallback
        return None


# ══════════════════════════════════════════════════════════════
# Main GPU Compute Class (Auto-selects Best Backend)
# ══════════════════════════════════════════════════════════════

class GPUCompute:
    """
    GPU-accelerated computation with automatic backend selection.
    
    Priority:
    1. CuPy/CUDA (NVIDIA GPUs - fastest)
    2. OpenCL (AMD/NVIDIA/Intel - portable)
    3. CPU fallback (no GPU)
    """
    
    def __init__(self):
        self.backend = None
        self.backend_name = "None"
        self.available = False
        self.device_name = "None" 
        self.platform_name = "None"
        self._init_error = None
        self.supports_fft = False
        
        # Try CuPy first (fastest for NVIDIA)
        if _CUPY_AVAILABLE:
            try:
                cupy_backend = CuPyBackend()
                if cupy_backend.available:
                    self.backend = cupy_backend
                    self.backend_name = "CuPy/CUDA"
                    self.available = True
                    self.device_name = cupy_backend.device_name
                    self.platform_name = cupy_backend.platform_name
                    self.supports_fft = True
                    return
            except Exception as e:
                pass
        
        # Fallback to OpenCL
        if _OPENCL_AVAILABLE:
            try:
                opencl_backend = OpenCLBackend()
                if opencl_backend.available:
                    self.backend = opencl_backend
                    self.backend_name = "OpenCL"
                    self.available = True
                    self.device_name = opencl_backend.device_name
                    self.platform_name = opencl_backend.platform_name
                    self.supports_fft = False  # OpenCL backend doesn't have FFT yet
                    return
            except Exception as e:
                pass
        
        # No GPU available
        self._init_error = "No GPU backend available (install cupy or pyopencl)"
    
    def status_string(self):
        """Return human-readable GPU status."""
        if self.available:
            fft_str = " [FFT+SCD]" if self.supports_fft else " [SCD only]"
            return f"{self.backend_name}: {self.device_name}{fft_str}"
        return f"GPU unavailable: {self._init_error}"
    
    def compute_scd_gpu(self, ffts_complex, m_vals, N):
        """Compute SCD on GPU."""
        if not self.available:
            raise RuntimeError("GPU not available")
        return self.backend.compute_scd_gpu(ffts_complex, m_vals, N)
    
    def compute_fft_batch_gpu(self, segments, window):
        """Batch FFT on GPU (returns None if not supported)."""
        if not self.available or not self.supports_fft:
            return None
        return self.backend.compute_fft_batch_gpu(segments, window)
