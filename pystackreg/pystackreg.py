# -*- coding: utf-8 -*-
from . import turboreg  # type: ignore
import numpy as np
from tqdm import tqdm
import warnings
from .util import simple_slice, running_mean


class StackReg:
    """
    Python implementation of the ImageJ/Fiji StackReg plugin
    (http://bigwww.epfl.ch/thevenaz/stackreg/ )
    """

    TRANSLATION = 2
    """
    Translation
    """

    RIGID_BODY = 3
    """
    Rigid body (translation + rotation)
    """

    SCALED_ROTATION = 4
    """
    Scaled rotation (translation + rotation + scaling)
    """

    AFFINE = 6
    """
    Affine (translation + rotation + scaling + shearing)
    """

    BILINEAR = 8
    """
    Bilinear (non-linear transformation; does not preserve straight lines)
    """

    _valid_transformations = [
        TRANSLATION,
        RIGID_BODY,
        SCALED_ROTATION,
        AFFINE,
        BILINEAR,
    ]

    _is_registered = False

    def __init__(self, transformation):
        """
        Constructor
        :param transformation: TRANSLATION, RIGID_BODY, SCALED_ROTATION, AFFINE,
                               BILINEAR
        """
        if transformation not in self._valid_transformations:
            raise Exception("Invalid transformation")

        self._transformation = transformation
        self._m = None
        self._tmats = None
        self._refpts = None
        self._movpts = None

    def is_registered(self):
        """
        Indicates whether register() was already called and a transformation matrix
        was calculated

        :rtype:  bool
        :return: True if a transformation matrix was already calculated
        """
        return self._is_registered

    def register(self, ref, mov, ref_mask=None, mov_mask=None):
        """
        Registers an image to a reference image: Only the transformation matrix
        will be calculated, the image will not be transformed
        (use transform() or register_transform() ).

        :type ref: array_like (Ni..., Nj...)
        :param ref: Reference image (static)

        :type mov: array_like (Ni..., Nj...)
        :param mov:
            The image that should be aligned to the reference image

        :type ref_mask: array_like (Ni..., Nj...), optional
        :param ref_mask:
            Optional mask for the reference image. Non-zero/positive values are
            included. If None, the full image is used (legacy behaviour).

        :type mov_mask: array_like (Ni..., Nj...), optional
        :param mov_mask:
            Optional mask for the moving image. Non-zero/positive values are
            included. If None, the full image is used (legacy behaviour).

        :rtype:  ndarray(3,3)
        :return: 2D transformation matrix
        """
        self._is_registered = True

        ref = np.asarray(ref, dtype=np.double)
        mov = np.asarray(mov, dtype=np.double)
        if ref.shape != mov.shape:
            raise ValueError(
                "Reference and moving images must have the same shape, got "
                "{} and {}".format(ref.shape, mov.shape)
            )

        ref_mask = self._validate_mask(ref_mask, ref.shape, "ref_mask")
        mov_mask = self._validate_mask(mov_mask, mov.shape, "mov_mask")

        # TurboReg crops the last row/column; keep masks aligned with that crop.
        self._m, self._refpts, self._movpts = turboreg._register(
            ref[:-1, :-1],
            mov[:-1, :-1],
            None if ref_mask is None else ref_mask[:-1, :-1],
            None if mov_mask is None else mov_mask[:-1, :-1],
            self._transformation,
        )

        return self.get_matrix()

    @staticmethod
    def _validate_mask(mask, shape, name):
        if mask is None:
            return None
        mask = np.asarray(mask, dtype=np.double)
        if mask.shape != shape:
            raise ValueError(
                "{} must have shape {}, got {}".format(name, shape, mask.shape)
            )
        return mask

    def transform(self, mov, tmat=None):
        """
        Transform an image according to a previous registration.

        Either a transformation matrix has to be explicitly supplied or
        register() has to be called before calling transform().

        :type mov: array_like (Ni..., Nj...)
        :param mov: The image that will be transformed

        :type tmat:  ndarray(3,3), optional
        :param tmat: The transformation matrix

        :rtype:  ndarray (Ni..., Nj...)
        :return: Transformed image - will be of the same shape as the input image
                 (cropping may occur)
        """
        if tmat is None and not self.is_registered():
            raise Exception("Register first")

        if tmat is not None:
            tmat = self._matrix_long_to_short(tmat)
        else:
            tmat = self._m

        return turboreg._transform(mov, tmat)

    def register_transform(self, ref, mov, ref_mask=None, mov_mask=None):
        """
        Register and transform an image to a reference image.

        :type ref: ref: array_like (Ni..., Nj...)
        :param ref: Reference image (static)

        :type mov: array_like (Ni..., Nj...)
        :param mov: The image that should be aligned to the reference image

        :type ref_mask: array_like (Ni..., Nj...), optional
        :param ref_mask: Optional mask for the reference image

        :type mov_mask: array_like (Ni..., Nj...), optional
        :param mov_mask: Optional mask for the moving image

        :rtype:  ndarray (Ni..., Nj...)
        :return: Transformed image - will be of the same shape as the input image
                 (cropping may occur)
        """
        self.register(ref, mov, ref_mask=ref_mask, mov_mask=mov_mask)
        return self.transform(mov)

    def get_matrix(self):
        """
        Get the current transformation matrix

        :rtype:  ndarray(3,3) or ndarray(4,4) for bilinear transformation
        :return: The transformation matrix
        """
        return self._matrix_short_to_long(self._m)

    def set_matrix(self, mat):
        """
        Sets the current transformation matrix

        :type mat: ndarray(3,3) or ndarray(4,4) for bilinear transformation
        :param mat: The transformation matrix
        """

        exp_shape = (4, 4) if self._transformation == self.BILINEAR else (3, 3)

        if not exp_shape == mat.shape:
            raise Exception(
                "Invalid shape of transformation matrix: Expected %s, got %s"
                % (str(exp_shape), str(mat.shape))
            )

        self._m = self._matrix_long_to_short(mat)

        self._is_registered = True

    def _matrix_short_to_long(self, m):
        """
        Converts the transformation matrix from the short form used by TurboReg to
        the canonical form from linear algebra.

        :type m: array_like
        :param m: TurboReg transformation matrix

        :rtype:  ndarray(3,3)
        :return: Canonical transformation matrix
        """

        if self._transformation == self.TRANSLATION:
            mat = np.identity(3).astype(np.double)
            mat[0:2, 2] = m[:, 0]
        elif self._transformation in [
            self.RIGID_BODY,
            self.SCALED_ROTATION,
            self.AFFINE,
        ]:
            mat = np.identity(3).astype(np.double)
            mat[0:2, :] = m[:, [1, 2, 0]]
        elif self._transformation == self.BILINEAR:
            mat = np.identity(4).astype(np.double)
            mat[0:2, :] = m[:, [1, 2, 3, 0]]
        else:
            raise Exception("Unexpected transformation")

        return mat

    def _matrix_long_to_short(self, mat):
        """
        Converts the transformation matrix from the canonical form from linear algebra
        to the short form used by TurboReg.

        :type mat: ndarray(3,3)
        :param mat: Canonical transformation matrix

        :rtype:  array_like
        :return: TurboReg transformation matrix
        """
        if self._transformation == self.TRANSLATION:
            m = mat[0:2, 2].reshape((2, 1))
        elif self._transformation in [
            self.RIGID_BODY,
            self.SCALED_ROTATION,
            self.AFFINE,
        ]:
            m = mat[0:2, [2, 0, 1]]
        elif self._transformation == self.BILINEAR:
            m = mat[0:2, [3, 0, 1, 2]]
        else:
            raise Exception("Unexpected transformation")

        return m.astype(np.double)

    @staticmethod
    def _detect_time_axis(img):
        """
        Detects the time axis of a movie/stack by returning the dimension with the
        lowest average variability

        :param img:
        :return:
        """
        var = np.array([np.var(img, axis=i).mean() for i in range(img.ndim)])
        return np.argmin(var)

    def get_points(self):
        """
        Returns the pairs of corresponding points from which the
        transformation matrix can be calculated.

        :rtype:  (ndarray, ndarray)
        :return: (Point coordinates of reference image, Point coordinates of image to
                 be aligned)
        """
        return self._refpts, self._movpts

    def register_stack(
        self,
        img,
        reference="previous",
        n_frames=1,
        axis=0,
        moving_average=1,
        verbose=False,
        progress_callback=None,
        suppress_axis_warning=False,
        mask=None,
    ):
        """
        Register a stack of images (movie).
        Note that this function will not transform the image but only calculate the
        transformation matrices.
        For transformation use transform_stack() after this function or use
        register_transform_stack() for a single call.

        :type img: array_like(Ni..., Nj..., Nk...)
        :param img: The 3D stack of images that should be aligned

        :type reference: string, one of ['previous', 'first', 'mean']
        :param reference:
            *  *'previous'*: Aligns each frame (image) to its previous frame in the
               stack
            *  *'first:'* Aligns each frame (image) to the first frame in the stack -
               if n_frames is > 1, then each frame is  aligned to the
               mean of the first n_frames of the stack
            *  *'mean'*: Aligns each frame (image) to the average of all images in the
               stack

        :type n_frames: int, optional
        :param n_frames: If reference is 'first', then this parameter specifies the
                         number of frames from the beginning of the stack that should
                         be averaged to yield the reference image.

        :type axis: int, optional
        :param axis: The axis of the time dimension

        :type moving_average: int, optional
        :param moving_average:
            If moving_average is greater than 1, a moving average of the stack is first
            created (using a subset size of moving_average) before registration

        :type verbose: bool, optional
        :param verbose:
            Specifies whether a progressbar should be shown using tqdm.

        :type progress_callback: function, optional
        :param progress_callback:
            A function that is called after every iteration. This function should accept
            the keyword arguments current_iteration:int and end_iteration:int.

        :type suppress_axis_warning: function, optional
        :param suppress_axis_warning:
            pystackreg will automatically try to determine the time axis and raise a
            warning when the detected time axis is not equal to the supplied axis.
            Set this option to True to suppress this warning.

        :type mask: array_like, optional
        :param mask:
            Optional registration mask. May be:

            * ``None``: no masking (legacy behaviour)
            * 2D array matching a single frame: same mask for every frame
            * 3D array matching ``img``: per-frame masks. For
              ``reference='previous'``, frame ``i-1`` masks the reference and
              frame ``i`` masks the moving image. For ``'first'`` / ``'mean'``,
              the reference mask is the mean of the corresponding frames.

        :rtype:  ndarray(img.shape[axis], 3, 3)
        :return: The transformation matrix for each image in the stack
        """

        if self._transformation == self.BILINEAR and reference == "previous":
            raise Exception(
                "Bilinear stack transformation not supported with reference "
                '== "previous" as a combination of bilinear transformations does not '
                "generally result in a bilinear transformation. Use another reference "
                "or manually register/transform images to their previous image."
            )

        if len(img.shape) != 3:
            raise Exception("Stack must have three dimensions")

        if not suppress_axis_warning:
            lowest_var_axis = self._detect_time_axis(img)
            if axis != lowest_var_axis:
                warnings.warn(
                    "Detected axis {} as the possible time axis for the stack due to "
                    "its low variability, but axis {} was supplied for registration. "
                    "Are you sure you supplied the correct axis?".format(
                        lowest_var_axis, axis
                    )
                )

        idx_start = 1

        if moving_average > 1:
            idx_start = 0
            size = [0] * len(img.shape)
            size[axis] = moving_average
            img = running_mean(img, moving_average, axis=axis)

        tmatdim = 4 if self._transformation == self.BILINEAR else 3

        self._tmats = np.repeat(
            np.identity(tmatdim).reshape((1, tmatdim, tmatdim)), img.shape[axis], axis=0
        ).astype(np.double)

        mask_stack = None
        static_mask = None
        if mask is not None:
            mask = np.asarray(mask, dtype=np.double)
            if mask.ndim == 2:
                frame_shape = tuple(
                    s for ax, s in enumerate(img.shape) if ax != axis
                )
                if mask.shape != frame_shape:
                    raise ValueError(
                        "2D mask must match frame shape {}, got {}".format(
                            frame_shape, mask.shape
                        )
                    )
                static_mask = mask
            elif mask.ndim == 3:
                if mask.shape != img.shape:
                    raise ValueError(
                        "3D mask must match image stack shape {}, got {}".format(
                            img.shape, mask.shape
                        )
                    )
                mask_stack = mask
            else:
                raise ValueError(
                    "mask must be None, a 2D frame mask, or a 3D stack mask"
                )

        if reference == "first":
            ref = np.mean(img.take(range(n_frames), axis=axis), axis=axis)
            if mask_stack is not None:
                ref_mask = np.mean(
                    mask_stack.take(range(n_frames), axis=axis), axis=axis
                )
            else:
                ref_mask = static_mask
        elif reference == "mean":
            ref = img.mean(axis=axis)
            idx_start = 0
            if mask_stack is not None:
                ref_mask = mask_stack.mean(axis=axis)
            else:
                ref_mask = static_mask
        elif reference == "previous":
            ref_mask = None
        else:
            raise ValueError('Unknown reference "{}"'.format(reference))

        iterable = range(idx_start, img.shape[axis])

        if verbose:
            iterable = tqdm(iterable)

        for i in iterable:
            slc = [slice(None)] * len(img.shape)
            slc[axis] = i

            if reference == "previous":
                ref = img.take(i - 1, axis=axis)
                if mask_stack is not None:
                    ref_mask = simple_slice(mask_stack, i - 1, axis)
                else:
                    ref_mask = static_mask

            if mask_stack is not None:
                mov_mask = simple_slice(mask_stack, i, axis)
            else:
                mov_mask = static_mask

            self._tmats[i, :, :] = self.register(
                ref,
                simple_slice(img, i, axis),
                ref_mask=ref_mask,
                mov_mask=mov_mask,
            )

            if reference == "previous" and i > 0:
                self._tmats[i, :, :] = np.matmul(
                    self._tmats[i, :, :], self._tmats[i - 1, :, :]
                )

            if progress_callback is not None:
                progress_callback(
                    current_iteration=i - idx_start + 1,
                    end_iteration=img.shape[axis] - idx_start,
                )

        return self._tmats

    def transform_stack(self, img, axis=0, tmats=None):
        """
        Transform a stack after registration.

        :type img: array_like(Ni..., Nj..., Nk...)
        :param img: The 3D stack of images that should be aligned

        :type axis: int, optional
        :param axis: The axis of the time dimension

        :type tmats: ndarray(img.shape[axis], 3, 3), optional
        :param tmats: The transformation matrix for each image in the stack

        :rtype:  ndarray(Ni..., Nj..., Nk...)
        :return: The transformed stack
        """
        if tmats is None:
            if self._tmats is None:
                raise Exception(
                    "No transformation matrices given. Please register first or pass "
                    "transformation matrices explicitly."
                )
            tmats = self._tmats

        if tmats.shape[0] != img.shape[axis]:
            raise Exception(
                "Number of saved transformation matrices does not match stack length"
            )

        out = img.copy().astype(float)

        for i in range(img.shape[axis]):
            slc = [slice(None)] * len(out.shape)
            slc[axis] = i
            out[tuple(slc)] = self.transform(simple_slice(img, i, axis), tmats[i, :, :])

        return out

    def register_transform_stack(
        self,
        img,
        reference="previous",
        n_frames=1,
        axis=0,
        moving_average=1,
        verbose=False,
        progress_callback=None,
        mask=None,
    ):
        """
        Register and transform stack of images (movie).

        :type img: array_like(Ni..., Nj..., Nk...)
        :param img: The 3D stack of images that should be aligned

        :type reference: string, one of ['previous', 'first', 'mean']
        :param reference:
            *  *'previous'*: Aligns each frame (image) to its previous frame in the
               stack
            *  *'first:'* Aligns each frame (image) to the first frame in the stack -
               if n_frames is > 1, then each frame is aligned to the
               mean of the first n_frames of the stack
            *  *'mean'*: Aligns each frame (image) to the average of all images in the
               stack

        :type n_frames: int, optional
        :param n_frames: If reference is 'first', then this parameter specifies the
                         number of frames from the beginning of the stack that
                         should be averaged to yield the reference image.

        :type axis: int, optional
        :param axis: The axis of the time dimension

        :type moving_average: int, optional
        :param moving_average: If moving_average is greater than 1, a moving average of
                               the stack is first created (using a subset size of
                               moving_average) before registration

        :type verbose: bool, optional
        :param verbose:
            Specifies whether a progressbar should be shown using tqdm.

        :type progress_callback: function, optional
        :param progress_callback:
            A function that is called after every iteration. This function should accept
            the keyword arguments current_iteration:int and end_iteration:int.

        :type mask: array_like, optional
        :param mask: Optional 2D or 3D registration mask (see register_stack)

        :rtype:  ndarray(Ni..., Nj..., Nk...)
        :return: The transformed stack
        """
        self.register_stack(
            img,
            reference=reference,
            n_frames=n_frames,
            axis=axis,
            moving_average=moving_average,
            verbose=verbose,
            progress_callback=progress_callback,
            mask=mask,
        )

        return self.transform_stack(img, axis)
