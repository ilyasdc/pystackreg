import os
import pytest
import numpy as np
import tifffile as tif

from pystackreg import (
    StackReg,
    StackRegCUDA,
    is_cuda_available,
    get_cuda_device_name,
    get_cuda_device_count,
)

BASE_PATH = os.path.dirname(__file__) + "/data/"
prefix = "pc12"


def np_school_round(x):
    x = np.array(x)
    r = x.copy()
    idx = x - np.floor(x) < 0.5
    r[idx] = np.floor(x[idx])
    r[~idx] = np.ceil(x[~idx])
    return r


def to_uint16(img):
    if hasattr(img, "get"):
        img = img.get()
    return np_school_round(img.clip(min=0, max=65535)).astype(np.uint16)


def load_file(transformation):
    if transformation == "unregistered":
        fname = prefix + "-unreg"
    else:
        fname = prefix + "-reg-" + transformation
    return tif.imread(os.path.join(BASE_PATH, fname + ".tif"))


@pytest.fixture
def stack_unregistered():
    return load_file("unregistered")


@pytest.fixture(
    params=[
        StackReg.TRANSLATION,
        StackReg.RIGID_BODY,
        StackReg.SCALED_ROTATION,
        StackReg.AFFINE,
        StackReg.BILINEAR,
    ]
)
def transformation_type(request):
    return request.param


# Ensure CUDA is available before running tests in this module
pytestmark = pytest.mark.skipif(
    not is_cuda_available(), reason="CUDA device or CuPy not available"
)


def test_cuda_device_info():
    """Verify CUDA device detection, name, and count."""
    assert is_cuda_available() is True
    count = get_cuda_device_count()
    assert count >= 1
    name = get_cuda_device_name(0)
    assert isinstance(name, str)
    assert len(name) > 0


def test_stackreg_devices():
    """Verify StackReg device parameter handling."""
    sr_auto = StackReg(StackReg.RIGID_BODY, device="auto")
    assert sr_auto.device == "cuda"

    sr_cuda = StackReg(StackReg.RIGID_BODY, device="cuda")
    assert sr_cuda.device == "cuda"

    sr_cpu = StackReg(StackReg.RIGID_BODY, device="cpu")
    assert sr_cpu.device == "cpu"

    with pytest.raises(ValueError):
        StackReg(StackReg.RIGID_BODY, device="tpu")


def test_stackreg_cuda_subclass():
    """Verify StackRegCUDA subclass defaults to device='cuda'."""
    sr = StackRegCUDA(StackReg.AFFINE)
    assert sr.device == "cuda"
    assert sr._transformation == StackReg.AFFINE


def test_cuda_2d_transform_against_cpu(transformation_type, stack_unregistered):
    """Verify CUDA 2D transform produces numerical parity with CPU TurboReg."""
    ref = stack_unregistered[0]
    mov = stack_unregistered[1]

    sr_cpu = StackReg(transformation_type, device="cpu")
    tmat = sr_cpu.register(ref, mov)

    # CPU warp
    out_cpu = sr_cpu.transform(mov)

    # CUDA warp
    sr_cuda = StackReg(transformation_type, device="cuda")
    sr_cuda.set_matrix(tmat)
    out_cuda = sr_cuda.transform(mov)

    # Max difference across all pixels must be <= 1 uint16 (exact parity with ImageJ)
    diff = np.abs(out_cuda - out_cpu)
    assert np.max(diff) <= 1.0


def test_cuda_3d_transform_stack_against_cpu(transformation_type, stack_unregistered):
    """Verify CUDA 3D batched transform_stack produces identical output to CPU."""
    reference = "first" if transformation_type == StackReg.BILINEAR else "previous"

    sr_cpu = StackReg(transformation_type, device="cpu")
    tmats = sr_cpu.register_stack(stack_unregistered, reference=reference, n_workers=1)
    out_cpu = sr_cpu.transform_stack(stack_unregistered)

    sr_cuda = StackRegCUDA(transformation_type)
    out_cuda = sr_cuda.transform_stack(stack_unregistered, tmats=tmats)

    diff = np.abs(out_cuda - out_cpu)
    assert np.max(diff) <= 1.0


def test_cuda_register_transform_stack_groundtruth(transformation_type, stack_unregistered):
    """Verify StackRegCUDA register_transform_stack matches ImageJ reference data."""
    trans_map = {
        StackReg.TRANSLATION: "translation",
        StackReg.RIGID_BODY: "rigid-body",
        StackReg.SCALED_ROTATION: "scaled-rotation",
        StackReg.AFFINE: "affine",
        StackReg.BILINEAR: "bilinear",
    }
    reference_data = load_file(trans_map[transformation_type])
    reference = "first" if transformation_type == StackReg.BILINEAR else "previous"

    sr = StackRegCUDA(transformation_type)
    out = sr.register_transform_stack(stack_unregistered, reference=reference)

    np.testing.assert_allclose(
        to_uint16(out), to_uint16(reference_data), rtol=1e-7, atol=1
    )


def test_cupy_array_inputs_and_outputs(stack_unregistered):
    """Verify seamless execution with CuPy arrays on GPU without CPU fallback."""
    import cupy as cp

    ref = stack_unregistered[0]
    mov = stack_unregistered[1]
    cp_mov = cp.asarray(mov, dtype=cp.float64)
    cp_stack = cp.asarray(stack_unregistered, dtype=cp.float64)

    sr = StackRegCUDA(StackReg.RIGID_BODY)

    # Register using cupy arrays
    tmat = sr.register(cp.asarray(ref), cp_mov)
    assert isinstance(tmat, np.ndarray)
    assert tmat.shape == (3, 3)

    # Transform 2D CuPy array -> CuPy array output
    cp_out_2d = sr.transform(cp_mov)
    assert isinstance(cp_out_2d, cp.ndarray)
    assert cp_out_2d.shape == mov.shape

    # Transform 3D CuPy stack -> CuPy array output
    tmats = sr.register_stack(cp_stack, reference="previous")
    cp_out_3d = sr.transform_stack(cp_stack, tmats=tmats)
    assert isinstance(cp_out_3d, cp.ndarray)
    assert cp_out_3d.shape == stack_unregistered.shape

    # Full register_transform_stack with CuPy stack
    cp_registered = sr.register_transform_stack(cp_stack, reference="previous")
    assert isinstance(cp_registered, cp.ndarray)
    assert cp_registered.shape == stack_unregistered.shape


def test_multithreading_workers(stack_unregistered):
    """Verify n_workers parameter produces identical results across thread counts."""
    sr1 = StackReg(StackReg.TRANSLATION, device="cuda")
    tmats_seq = sr1.register_stack(stack_unregistered, reference="previous", n_workers=1)

    sr4 = StackReg(StackReg.TRANSLATION, device="cuda")
    tmats_par = sr4.register_stack(stack_unregistered, reference="previous", n_workers=4)

    sr_auto = StackReg(StackReg.TRANSLATION, device="cuda")
    tmats_auto = sr_auto.register_stack(stack_unregistered, reference="previous", n_workers=None)

    np.testing.assert_allclose(tmats_seq, tmats_par, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(tmats_seq, tmats_auto, rtol=1e-10, atol=1e-10)


def test_cuda_masking(stack_unregistered):
    """Verify masked registration and transformation on CUDA."""
    ref = stack_unregistered[0]
    mov = stack_unregistered[1]
    mask = np.ones_like(ref, dtype=np.float64)
    mask[:25, :25] = 0

    sr_cpu = StackReg(StackReg.RIGID_BODY, device="cpu")
    tmat_cpu = sr_cpu.register(ref, mov, ref_mask=mask, mov_mask=mask)

    sr_cuda = StackRegCUDA(StackReg.RIGID_BODY)
    tmat_cuda = sr_cuda.register(ref, mov, ref_mask=mask, mov_mask=mask)

    np.testing.assert_allclose(tmat_cpu, tmat_cuda, rtol=1e-10, atol=1e-10)

    # Transform on GPU
    out_cuda = sr_cuda.transform(mov)
    assert out_cuda.shape == mov.shape


@pytest.mark.parametrize("frame_axis", [1, 2])
def test_cuda_different_axes(stack_unregistered, frame_axis):
    """Verify CUDA transform_stack and register_transform_stack along non-zero axes."""
    stack_axis = np.moveaxis(stack_unregistered, 0, frame_axis)

    sr = StackRegCUDA(StackReg.TRANSLATION)
    out = sr.register_transform_stack(stack_axis, reference="previous", axis=frame_axis)

    assert out.shape == stack_axis.shape
    # Check that when transformed back, it matches standard axis=0 registration
    out_standard = np.moveaxis(out, frame_axis, 0)
    ref_sr = StackRegCUDA(StackReg.TRANSLATION)
    out_ref = ref_sr.register_transform_stack(stack_unregistered, reference="previous", axis=0)

    diff = np.abs(out_standard - out_ref)
    assert np.max(diff) <= 1.0


def test_pytorch_support_if_available(stack_unregistered):
    """Verify PyTorch tensor support when PyTorch with CUDA is available."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch installed but CUDA is not available in PyTorch")

    t_mov = torch.from_numpy(stack_unregistered[1].astype(np.float64)).cuda()
    t_stack = torch.from_numpy(stack_unregistered.astype(np.float64)).cuda()

    sr = StackRegCUDA(StackReg.RIGID_BODY)
    sr.register(stack_unregistered[0], stack_unregistered[1])

    # 2D transform
    out_2d = sr.transform(t_mov)
    assert isinstance(out_2d, torch.Tensor)
    assert out_2d.is_cuda
    assert out_2d.shape == t_mov.shape

    # 3D transform stack
    out_3d = sr.transform_stack(t_stack)
    assert isinstance(out_3d, torch.Tensor)
    assert out_3d.is_cuda
    assert out_3d.shape == t_stack.shape

