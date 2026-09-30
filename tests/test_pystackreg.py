import os
import pytest
import numpy as np
from pystackreg import StackReg
import tifffile as tif

BASE_PATH = os.path.dirname(__file__) + "/data/"
prefix = "pc12"


def np_school_round(x):
    """
    Rounding function that always round .5 value up (as opposed to standard
    python/numpy behaviour, which round .5 values to the next even number.
    This function here mimics the behavior of ImageJ for comparability of
    StackReg/TurboReg registrations from ImageJ and pystackreg.

    :param x: (float) number
    :return: (float) rounded number (.5 always rounded up)
    """
    x = np.array(x)
    r = x.copy()
    idx = x - np.floor(x) < 0.5
    r[idx] = np.floor(x[idx])
    r[~idx] = np.ceil(x[~idx])

    return r


def to_uint16(img):
    return np_school_round(img.clip(min=0, max=65535)).astype(np.uint16)


@pytest.fixture
def stack_unregistered():
    return load_file("unregistered")


@pytest.fixture
def stack_translation():
    return load_file("translation")


@pytest.fixture
def stack_rigid_body():
    return load_file("rigid-body")


@pytest.fixture
def stack_scaled_roation():
    return load_file("scaled-rotation")


@pytest.fixture
def stack_affine():
    return load_file("affine")


@pytest.fixture
def stack_bilinear():
    return load_file("bilinear")


def load_file(transformation):
    if transformation == "unregistered":
        fname = prefix + "-unreg"
    else:
        fname = prefix + "-reg-" + transformation

    return tif.imread(os.path.join(BASE_PATH, fname + ".tif"))


@pytest.fixture(
    params=[
        StackReg.TRANSLATION,
        StackReg.RIGID_BODY,
        StackReg.SCALED_ROTATION,
        StackReg.AFFINE,
        StackReg.BILINEAR,
    ]
)
def stack(request):
    transformation_map = {
        StackReg.TRANSLATION: "translation",
        StackReg.RIGID_BODY: "rigid-body",
        StackReg.SCALED_ROTATION: "scaled-rotation",
        StackReg.AFFINE: "affine",
        StackReg.BILINEAR: "bilinear",
    }

    return {
        "transformation": request.param,
        "registered": load_file(transformation_map[request.param]),
    }


def test_registration_transformation(stack, stack_unregistered):
    sr = StackReg(stack["transformation"])
    reference = (
        "previous" if not stack["transformation"] == StackReg.BILINEAR else "first"
    )
    out = sr.register_transform_stack(stack_unregistered, reference=reference)

    np.testing.assert_allclose(
        to_uint16(out), to_uint16(stack["registered"]), rtol=1e-7, atol=1
    )


def test_optional_mask_none_matches_unmasked(stack_unregistered):
    """Omitting masks must match legacy (fully unmasked) registration."""
    sr = StackReg(StackReg.TRANSLATION)
    tmats_a = sr.register_stack(stack_unregistered, reference="previous")
    tmats_b = sr.register_stack(stack_unregistered, reference="previous", mask=None)
    np.testing.assert_allclose(tmats_a, tmats_b)

    ones = np.ones(stack_unregistered.shape[1:], dtype=np.float64)
    tmats_c = sr.register_stack(stack_unregistered, reference="previous", mask=ones)
    np.testing.assert_allclose(tmats_a, tmats_c, rtol=1e-10, atol=1e-10)


def test_register_with_pair_masks(stack_unregistered):
    ref = stack_unregistered[0]
    mov = stack_unregistered[1]
    sr = StackReg(StackReg.TRANSLATION)

    m_none = sr.register(ref, mov)
    m_ones = sr.register(
        ref, mov, ref_mask=np.ones_like(ref), mov_mask=np.ones_like(mov)
    )
    np.testing.assert_allclose(m_none, m_ones, rtol=1e-10, atol=1e-10)

    # Zeroing out a corner should still produce a finite transform.
    mask = np.ones_like(ref)
    mask[:20, :20] = 0
    m_masked = sr.register(ref, mov, ref_mask=mask, mov_mask=mask)
    assert m_masked.shape == (3, 3)
    assert np.all(np.isfinite(m_masked))


def test_stack_mask_shapes(stack_unregistered):
    sr = StackReg(StackReg.TRANSLATION)
    frame_mask = np.ones(stack_unregistered.shape[1:], dtype=np.float64)
    stack_mask = np.ones_like(stack_unregistered, dtype=np.float64)

    tmats_2d = sr.register_stack(
        stack_unregistered, reference="first", mask=frame_mask
    )
    tmats_3d = sr.register_stack(
        stack_unregistered, reference="first", mask=stack_mask
    )
    np.testing.assert_allclose(tmats_2d, tmats_3d, rtol=1e-10, atol=1e-10)

    with pytest.raises(ValueError):
        sr.register_stack(
            stack_unregistered, reference="first", mask=np.ones((3, 3))
        )


@pytest.mark.parametrize("frame_axis", [1, 2])
def test_different_axis(stack, stack_unregistered, frame_axis):
    stack["registered"] = np.moveaxis(stack["registered"], 0, frame_axis)
    stack_unregistered = np.moveaxis(stack_unregistered, 0, frame_axis)

    sr = StackReg(stack["transformation"])
    reference = (
        "previous" if not stack["transformation"] == StackReg.BILINEAR else "first"
    )
    out = sr.register_transform_stack(
        stack_unregistered, reference=reference, axis=frame_axis
    )

    assert out.shape == stack["registered"].shape
    np.testing.assert_allclose(
        to_uint16(out),
        to_uint16(stack["registered"]),
        rtol=1e-7,
        atol=1,
    )
