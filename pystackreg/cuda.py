# -*- coding: utf-8 -*-
"""
CUDA backend for PyStackReg.

Provides GPU-accelerated image transformation and registration using NVIDIA CUDA.
Supports NumPy arrays, CuPy arrays, and PyTorch CUDA tensors directly without
unnecessary host-device copies when data already resides on the GPU.
"""

import os
import sys
import tempfile
import numpy as np

# Ensure CuPy disk cache directory is writable (handles sandboxed environments)
if "CUPY_CACHE_DIR" not in os.environ:
    try:
        home_cupy = os.path.expanduser("~/.cupy")
        os.makedirs(home_cupy, exist_ok=True)
    except OSError:
        os.environ["CUPY_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "cupy_cache")

# Auto-detect CUDA headers in Python virtualenv (e.g. nvidia-cuda-runtime-cu12)
if "CUDA_PATH" not in os.environ:
    try:
        import nvidia.cuda_runtime

        cuda_runtime_dir = os.path.dirname(nvidia.cuda_runtime.__file__)
        if os.path.isdir(os.path.join(cuda_runtime_dir, "include")):
            os.environ["CUDA_PATH"] = cuda_runtime_dir
    except Exception:
        pass

# Global references to compiled CUDA kernels and CuPy module
_cupy = None
_cuda_available = None
_cuda_error = None
_kernels = {}


def is_cuda_available() -> bool:
    """
    Check if CUDA and CuPy are available and operational.

    :return: True if an NVIDIA GPU is available and CuPy can initialize it.
    """
    global _cupy, _cuda_available, _cuda_error
    if _cuda_available is not None:
        return _cuda_available

    try:
        import cupy as cp

        # Attempt small allocation on default device to verify runtime
        props = cp.cuda.runtime.getDeviceProperties(0)
        _cupy = cp
        _cuda_available = True
        _cuda_error = None
        return True
    except Exception as e:
        _cuda_available = False
        _cuda_error = str(e)
        return False


def get_cuda_device_name(device_id: int = 0) -> str:
    """Return the name of the CUDA device."""
    if not is_cuda_available():
        raise RuntimeError(f"CUDA is not available: {_cuda_error}")
    props = _cupy.cuda.runtime.getDeviceProperties(device_id)
    return props["name"].decode("utf-8")


def get_cuda_device_count() -> int:
    """Return the number of CUDA devices available."""
    if not is_cuda_available():
        return 0
    return _cupy.cuda.runtime.getDeviceCount()


def _get_cuda_source() -> str:
    """Load the CUDA source code for TurboReg kernels."""
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "turboreg_cuda.cu"),
        os.path.join(os.path.dirname(__file__), "..", "src", "turboreg_cuda.cu"),
    ]
    for p in candidate_paths:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            with open(p, "r") as f:
                return f.read()

    # Fallback embedded CUDA kernels if source file is missing
    return r"""
extern "C" {

#define BSPLINE_Z -0.26794919243112270647255365849413
#define BSPLINE_LAMBDA 6.0

__global__ void bspline_coeff_rows(double* __restrict__ img, int width, int height) {
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    if (y >= height) return;
    double* row = img + (long long)y * width;
    if (width <= 1) return;
    const double z = BSPLINE_Z;
    for (int n = 0; n < width; n++) row[n] *= BSPLINE_LAMBDA;
    double zn = pow(z, (double)width);
    double sum = (1.0 + z) * (row[0] + zn * row[width - 1]);
    double z1 = z, zn2 = zn * zn;
    for (int n = 1; n < width - 1; n++) {
        z1 *= z; zn2 /= z;
        sum += (z1 + zn2) * row[n];
    }
    row[0] = sum / (1.0 - pow(z, (double)(2 * width)));
    for (int n = 1; n < width; n++) row[n] += z * row[n - 1];
    row[width - 1] = z * row[width - 1] / (z - 1.0);
    for (int n = width - 2; n >= 0; n--) row[n] = z * (row[n + 1] - row[n]);
}

__global__ void bspline_coeff_cols(double* __restrict__ img, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= width || height <= 1) return;
    const double z = BSPLINE_Z;
    for (int n = 0; n < height; n++) img[(long long)n * width + x] *= BSPLINE_LAMBDA;
    double zn = pow(z, (double)height);
    double sum = (1.0 + z) * (img[x] + zn * img[(long long)(height - 1) * width + x]);
    double z1 = z, zn2 = zn * zn;
    for (int n = 1; n < height - 1; n++) {
        z1 *= z; zn2 /= z;
        sum += (z1 + zn2) * img[(long long)n * width + x];
    }
    img[x] = sum / (1.0 - pow(z, (double)(2 * height)));
    for (int n = 1; n < height; n++) img[(long long)n * width + x] += z * img[(long long)(n - 1) * width + x];
    img[(long long)(height - 1) * width + x] = z * img[(long long)(height - 1) * width + x] / (z - 1.0);
    for (int n = height - 2; n >= 0; n--) img[(long long)n * width + x] = z * (img[(long long)(n + 1) * width + x] - img[(long long)n * width + x]);
}

__global__ void bspline_coeff_3d_rows(double* __restrict__ stack, int width, int height, int numFrames) {
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int f = blockIdx.y;
    if (y >= height || f >= numFrames) return;
    double* row = stack + ((long long)f * height + y) * width;
    if (width <= 1) return;
    const double z = BSPLINE_Z;
    for (int n = 0; n < width; n++) row[n] *= BSPLINE_LAMBDA;
    double zn = pow(z, (double)width);
    double sum = (1.0 + z) * (row[0] + zn * row[width - 1]);
    double z1 = z, zn2 = zn * zn;
    for (int n = 1; n < width - 1; n++) {
        z1 *= z; zn2 /= z;
        sum += (z1 + zn2) * row[n];
    }
    row[0] = sum / (1.0 - pow(z, (double)(2 * width)));
    for (int n = 1; n < width; n++) row[n] += z * row[n - 1];
    row[width - 1] = z * row[width - 1] / (z - 1.0);
    for (int n = width - 2; n >= 0; n--) row[n] = z * (row[n + 1] - row[n]);
}

__global__ void bspline_coeff_3d_cols(double* __restrict__ stack, int width, int height, int numFrames) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int f = blockIdx.y;
    if (x >= width || f >= numFrames) return;
    if (height <= 1) return;
    double* frame = stack + (long long)f * height * width;
    const double z = BSPLINE_Z;
    for (int n = 0; n < height; n++) frame[(long long)n * width + x] *= BSPLINE_LAMBDA;
    double zn = pow(z, (double)height);
    double sum = (1.0 + z) * (frame[x] + zn * frame[(long long)(height - 1) * width + x]);
    double z1 = z, zn2 = zn * zn;
    for (int n = 1; n < height - 1; n++) {
        z1 *= z; zn2 /= z;
        sum += (z1 + zn2) * frame[(long long)n * width + x];
    }
    frame[x] = sum / (1.0 - pow(z, (double)(2 * height)));
    for (int n = 1; n < height; n++) frame[(long long)n * width + x] += z * frame[(long long)(n - 1) * width + x];
    frame[(long long)(height - 1) * width + x] = z * frame[(long long)(height - 1) * width + x] / (z - 1.0);
    for (int n = height - 2; n >= 0; n--) frame[(long long)n * width + x] = z * (frame[(long long)(n + 1) * width + x] - frame[(long long)n * width + x]);
}

__global__ void bspline_transform_2d(
    const double* __restrict__ inCoeff,
    double* __restrict__ outImg,
    int inNx, int inNy,
    int outNx, int outNy,
    double c0, double c1, double c2, double c3,
    double c4, double c5, double c6, double c7
) {
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    int v = blockDim.y * blockIdx.y + threadIdx.y;
    if (u >= outNx || v >= outNy) return;

    double x = c0 + c1 * u + c2 * v + c3 * u * v;
    double y = c4 + c5 * u + c6 * v + c7 * u * v;

    int xMsk = (0.0 <= x) ? ((int)(x + 0.5)) : ((int)(x - 0.5));
    int yMsk = (0.0 <= y) ? ((int)(y + 0.5)) : ((int)(y - 0.5));

    long long outIdx = (long long)v * outNx + u;
    if ((0 <= xMsk) && (xMsk < inNx) && (0 <= yMsk) && (yMsk < inNy)) {
        int twiceInNx = 2 * inNx;
        int twiceInNy = 2 * inNy;

        int px = (0.0 <= x) ? ((int)x + 2) : ((int)x + 1);
        int xIndex[4];
        for (int k = 0; k < 4; px--, k++) {
            int q = (px < 0) ? (-1 - px) : px;
            if (twiceInNx <= q) q -= twiceInNx * (q / twiceInNx);
            xIndex[k] = (inNx <= q) ? (twiceInNx - 1 - q) : q;
        }

        double rx = x - ((0.0 <= x) ? ((int)x) : ((int)x - 1));
        double sx = 1.0 - rx;
        double sxx = rx * rx;
        double xWeight[4];
        xWeight[3] = sx * sx * sx / 6.0;
        xWeight[2] = 2.0 / 3.0 - 0.5 * sxx * (2.0 - rx);
        xWeight[0] = sxx * rx / 6.0;
        xWeight[1] = 1.0 - xWeight[0] - xWeight[2] - xWeight[3];

        int py = (0.0 <= y) ? ((int)y + 2) : ((int)y + 1);
        int yIndex[4];
        for (int k = 0; k < 4; py--, k++) {
            int q = (py < 0) ? (-1 - py) : py;
            if (twiceInNy <= q) q -= twiceInNy * (q / twiceInNy);
            yIndex[k] = (inNy <= q) ? (twiceInNy - 1 - q) : q;
        }

        double ry = y - ((0.0 <= y) ? ((int)y) : ((int)y - 1));
        double sy = 1.0 - ry;
        double syy = ry * ry;
        double yWeight[4];
        yWeight[3] = sy * sy * sy / 6.0;
        yWeight[2] = 2.0 / 3.0 - 0.5 * syy * (2.0 - ry);
        yWeight[0] = syy * ry / 6.0;
        yWeight[1] = 1.0 - yWeight[0] - yWeight[2] - yWeight[3];

        double t = 0.0;
        for (int j = 0; j < 4; j++) {
            double s = 0.0;
            long long p = (long long)yIndex[j] * inNx;
            for (int i = 0; i < 4; i++) {
                s += xWeight[i] * inCoeff[p + xIndex[i]];
            }
            t += yWeight[j] * s;
        }
        outImg[outIdx] = (float)t;
    } else {
        outImg[outIdx] = 0.0;
    }
}

__global__ void bspline_transform_3d(
    const double* __restrict__ inCoeffs,
    double* __restrict__ outImgs,
    const double* __restrict__ params,
    int inNx, int inNy,
    int outNx, int outNy,
    int numFrames
) {
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    int v = blockDim.y * blockIdx.y + threadIdx.y;
    int f = blockIdx.z;

    if (u >= outNx || v >= outNy || f >= numFrames) return;

    const double* p = params + f * 8;
    double c0 = p[0], c1 = p[1], c2 = p[2], c3 = p[3];
    double c4 = p[4], c5 = p[5], c6 = p[6], c7 = p[7];

    double x = c0 + c1 * u + c2 * v + c3 * u * v;
    double y = c4 + c5 * u + c6 * v + c7 * u * v;

    int xMsk = (0.0 <= x) ? ((int)(x + 0.5)) : ((int)(x - 0.5));
    int yMsk = (0.0 <= y) ? ((int)(y + 0.5)) : ((int)(y - 0.5));

    long long frameOffset = (long long)f * inNx * inNy;
    long long outFrameOffset = (long long)f * outNx * outNy;
    long long outIdx = outFrameOffset + (long long)v * outNx + u;

    if ((0 <= xMsk) && (xMsk < inNx) && (0 <= yMsk) && (yMsk < inNy)) {
        int twiceInNx = 2 * inNx;
        int twiceInNy = 2 * inNy;

        int px = (0.0 <= x) ? ((int)x + 2) : ((int)x + 1);
        int xIndex[4];
        for (int k = 0; k < 4; px--, k++) {
            int q = (px < 0) ? (-1 - px) : px;
            if (twiceInNx <= q) q -= twiceInNx * (q / twiceInNx);
            xIndex[k] = (inNx <= q) ? (twiceInNx - 1 - q) : q;
        }

        double rx = x - ((0.0 <= x) ? ((int)x) : ((int)x - 1));
        double sx = 1.0 - rx;
        double sxx = rx * rx;
        double xWeight[4];
        xWeight[3] = sx * sx * sx / 6.0;
        xWeight[2] = 2.0 / 3.0 - 0.5 * sxx * (2.0 - rx);
        xWeight[0] = sxx * rx / 6.0;
        xWeight[1] = 1.0 - xWeight[0] - xWeight[2] - xWeight[3];

        int py = (0.0 <= y) ? ((int)y + 2) : ((int)y + 1);
        int yIndex[4];
        for (int k = 0; k < 4; py--, k++) {
            int q = (py < 0) ? (-1 - py) : py;
            if (twiceInNy <= q) q -= twiceInNy * (q / twiceInNy);
            yIndex[k] = (inNy <= q) ? (twiceInNy - 1 - q) : q;
        }

        double ry = y - ((0.0 <= y) ? ((int)y) : ((int)y - 1));
        double sy = 1.0 - ry;
        double syy = ry * ry;
        double yWeight[4];
        yWeight[3] = sy * sy * sy / 6.0;
        yWeight[2] = 2.0 / 3.0 - 0.5 * syy * (2.0 - ry);
        yWeight[0] = syy * ry / 6.0;
        yWeight[1] = 1.0 - yWeight[0] - yWeight[2] - yWeight[3];

        const double* inCoeff = inCoeffs + frameOffset;
        double t = 0.0;
        for (int j = 0; j < 4; j++) {
            double s = 0.0;
            long long rowIdx = (long long)yIndex[j] * inNx;
            for (int i = 0; i < 4; i++) {
                s += xWeight[i] * inCoeff[rowIdx + xIndex[i]];
            }
            t += yWeight[j] * s;
        }
        outImgs[outIdx] = (float)t;
    } else {
        outImgs[outIdx] = 0.0;
    }
}

} // extern "C"
"""


def _get_kernel(name: str):
    """Retrieve or lazily compile a CUDA kernel."""
    if not is_cuda_available():
        raise RuntimeError(f"CUDA is not available: {_cuda_error}")
    if name not in _kernels:
        src = _get_cuda_source()
        _kernels[name] = _cupy.RawKernel(src, name)
    return _kernels[name]


def short_matrix_to_params(m) -> tuple:
    """
    Convert a TurboReg short transformation matrix to the 8 canonical parameters
    (c0..c7) used by the CUDA cubic B-spline warp kernel:
      x = c0 + c1 * u + c2 * v + c3 * u * v
      y = c4 + c5 * u + c6 * v + c7 * u * v
    """
    m = np.asarray(m, dtype=np.float64)
    if m.shape == (2, 1):
        # Translation: x = m[0, 0] + u, y = m[1, 0] + v
        return (float(m[0, 0]), 1.0, 0.0, 0.0, float(m[1, 0]), 0.0, 1.0, 0.0)
    elif m.shape == (2, 3):
        # Affine / Rigid / Scaled rotation
        return (
            float(m[0, 0]),
            float(m[0, 1]),
            float(m[0, 2]),
            0.0,
            float(m[1, 0]),
            float(m[1, 1]),
            float(m[1, 2]),
            0.0,
        )
    elif m.shape == (2, 4):
        # Bilinear
        return (
            float(m[0, 0]),
            float(m[0, 1]),
            float(m[0, 2]),
            float(m[0, 3]),
            float(m[1, 0]),
            float(m[1, 1]),
            float(m[1, 2]),
            float(m[1, 3]),
        )
    else:
        raise ValueError(
            f"Invalid transformation matrix shape for TurboReg: {m.shape}"
        )


def _prepare_input_array(arr):
    """
    Inspect input array format and convert to CuPy float64 array.
    Returns (cp_arr, input_type, original_metadata).
    Supported types: 'numpy', 'cupy', 'torch'.
    """
    # Check for PyTorch tensor
    is_torch = False
    torch_device = None
    if "torch" in sys.modules or (hasattr(arr, "detach") and hasattr(arr, "is_cuda")):
        try:
            import torch

            if isinstance(arr, torch.Tensor):
                is_torch = True
                torch_device = arr.device
        except ImportError:
            pass

    if is_torch:
        if not is_cuda_available():
            raise RuntimeError(f"CUDA is not available: {_cuda_error}")
        # Zero-copy conversion between PyTorch and CuPy via __cuda_array_interface__
        cp_arr = _cupy.asarray(arr.detach(), dtype=_cupy.float64)
        return cp_arr, "torch", {"device": torch_device}

    # Check for CuPy array
    if is_cuda_available() and isinstance(arr, _cupy.ndarray):
        cp_arr = arr.astype(_cupy.float64, copy=False)
        return cp_arr, "cupy", None

    # NumPy / host array
    if not is_cuda_available():
        raise RuntimeError(f"CUDA is not available: {_cuda_error}")
    np_arr = np.asarray(arr, dtype=np.float64)
    cp_arr = _cupy.asarray(np_arr)
    return cp_arr, "numpy", None


def _format_output_array(cp_out, input_type, meta):
    """Convert CuPy output array back to the matching input type."""
    if input_type == "cupy":
        return cp_out
    elif input_type == "torch":
        import torch

        device = meta.get("device", None) if meta else None
        try:
            t = torch.as_tensor(cp_out)
        except Exception:
            t = torch.from_dlpack(cp_out.toDlpack())
        if device is not None:
            t = t.to(device)
        return t
    else:
        # Default numpy
        return _cupy.asnumpy(cp_out)


def bspline_coeff_2d(img):
    """
    Compute cubic B-spline interpolation coefficients for a 2D image on GPU.
    Modifies img in place if it's already a contiguous CuPy float64 array,
    or returns a new CuPy array.
    """
    cp = _cupy
    k_rows = _get_kernel("bspline_coeff_rows")
    k_cols = _get_kernel("bspline_coeff_cols")

    h, w = img.shape
    threads = 128
    grid_y = (h + threads - 1) // threads
    grid_x = (w + threads - 1) // threads

    k_rows((grid_y,), (threads,), (img, w, h))
    k_cols((grid_x,), (threads,), (img, w, h))
    return img


def bspline_coeff_3d(stack):
    """
    Compute cubic B-spline interpolation coefficients for a 3D movie stack on GPU.
    Operates across all frames in parallel.
    stack shape: (N, H, W).
    """
    cp = _cupy
    k_rows_3d = _get_kernel("bspline_coeff_3d_rows")
    k_cols_3d = _get_kernel("bspline_coeff_3d_cols")

    num_frames, h, w = stack.shape
    threads = 128
    grid_y = ((h + threads - 1) // threads, num_frames)
    grid_x = ((w + threads - 1) // threads, num_frames)

    k_rows_3d(grid_y, (threads,), (stack, w, h, num_frames))
    k_cols_3d(grid_x, (threads,), (stack, w, h, num_frames))
    return stack


def transform_cuda(mov, tmat_short):
    """
    Transform a 2D image according to a transformation matrix on CUDA GPU.

    :param mov: 2D image (NumPy array, CuPy array, or PyTorch CUDA tensor).
    :param tmat_short: TurboReg short-form transformation matrix.
    :return: Transformed 2D image matching the input array type.
    """
    if not is_cuda_available():
        raise RuntimeError(f"CUDA is not available: {_cuda_error}")

    cp = _cupy
    cp_mov, in_type, meta = _prepare_input_array(mov)
    if cp_mov.ndim != 2:
        raise ValueError(f"Input image must be 2D, got shape {cp_mov.shape}")

    h, w = cp_mov.shape
    # Compute B-spline coefficients on GPU
    d_coeff = cp_mov.copy()
    bspline_coeff_2d(d_coeff)

    # Output array
    d_out = cp.zeros((h, w), dtype=cp.float64)

    # Parameters
    p = short_matrix_to_params(tmat_short)

    k_trans = _get_kernel("bspline_transform_2d")
    block = (16, 16)
    grid = ((w + 15) // 16, (h + 15) // 16)
    k_trans(grid, block, (d_coeff, d_out, w, h, w, h, *p))

    return _format_output_array(d_out, in_type, meta)


def transform_stack_cuda(stack, tmats_short, axis=0):
    """
    Transform a 3D movie stack using a batched CUDA kernel across all frames.

    :param stack: 3D stack (NumPy array, CuPy array, or PyTorch CUDA tensor).
    :param tmats_short: Sequence or array of short-form transformation matrices.
    :param axis: Time axis (default 0).
    :return: Transformed 3D stack matching the input array type.
    """
    if not is_cuda_available():
        raise RuntimeError(f"CUDA is not available: {_cuda_error}")

    cp = _cupy
    cp_stack, in_type, meta = _prepare_input_array(stack)
    if cp_stack.ndim != 3:
        raise ValueError(f"Stack must be 3D, got shape {cp_stack.shape}")

    # Move axis to 0 if needed
    transposed = False
    if axis != 0:
        cp_stack = cp.moveaxis(cp_stack, axis, 0)
        transposed = True

    num_frames, h, w = cp_stack.shape
    if len(tmats_short) != num_frames:
        raise ValueError(
            f"Number of matrices ({len(tmats_short)}) does not match stack frames ({num_frames})"
        )

    # Prepare params matrix (N, 8)
    params_host = np.zeros((num_frames, 8), dtype=np.float64)
    for f in range(num_frames):
        params_host[f, :] = short_matrix_to_params(tmats_short[f])
    d_params = cp.asarray(params_host)

    # Compute B-spline coefficients for all frames in parallel on GPU
    d_coeff = cp_stack.copy()
    bspline_coeff_3d(d_coeff)

    # Output buffer
    d_out = cp.zeros((num_frames, h, w), dtype=cp.float64)

    # Launch 3D batched transformation kernel
    k_trans_3d = _get_kernel("bspline_transform_3d")
    block = (16, 16, 1)
    grid = ((w + 15) // 16, (h + 15) // 16, num_frames)
    k_trans_3d(grid, block, (d_coeff, d_out, d_params, w, h, w, h, num_frames))

    if transposed:
        d_out = cp.moveaxis(d_out, 0, axis)

    return _format_output_array(d_out, in_type, meta)
