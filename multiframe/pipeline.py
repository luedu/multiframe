import numpy as np
import cv2
import sys
from .utils import smooth_gradient

EPS = sys.float_info.epsilon


def denoising(image_stack: np.ndarray, pattern: str, motion_comp: bool = False, bitdepth: int = 8,
              max_global_motion: int = 30) -> np.array:
    """
    Denoising using a multiframe approach.

    Args:
        image_stack: Stack of images of size H x W x NumImages.
        pattern: Bayer pattern, one of {'rggb, 'grbg', 'gbrg', 'bggr'}.
        motion_comp: Toggle to activate motion compensation using the Median Threshold Bitmap (MTB) algorithm.
        bitdepth: RAW image bitdepth.
        max_global_motion: Maximum motion displacement allowed for correction.

    Returns:
        Denoised RAW image of size H x W.
    """
    if image_stack.ndim == 2:
        return image_stack

    im_type = image_stack.dtype

    # Motion compensation switch
    if not motion_comp:
        return image_stack.astype(np.float32).mean(axis=2).astype(im_type)

    else:
        max_pix_value = (2 ** bitdepth - 1)
        scaling = 255. / max_pix_value

        # Converted into 8-bit format to be able to use OpenCV
        image_stack = (image_stack.astype(np.float32) * scaling).astype(np.uint8)

        # Find the indices of the red and blue channels from the Bayer pattern
        idx = {c: [0, 0] for c in ['r', 'b', 'g1', 'g2']}
        n = 1
        for c, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
            if c != 'g':
                idx[c] = [y, x]
            else:
                idx[c + str(n)] = [y, x]
                n += 1

        # Global alignment using median threshold bitmaps (MTB)
        # For now we can only correct misalignments of an even number of pixels, as otherwise the Bayer pattern would
        # not match anymore
        alignment = cv2.createAlignMTB()
        shift = [None] * (image_stack.shape[2] - 1)
        image_stack_aligned = np.zeros_like(image_stack)
        ref_frame_idx = image_stack.shape[2] // 2  # Middle frame
        image_stack_aligned[:, :, ref_frame_idx] = image_stack[:, :, ref_frame_idx].copy()

        # Align the frames w.r.t. the reference frame using one of the green channels
        n = 0
        for i in range(image_stack.shape[2]):
            # Skip the reference frame (no correction)
            if i == ref_frame_idx:
                continue

            image_stack_aligned[:, :, i] = np.zeros_like(image_stack[:, :, 0])
            shift[n] = np.array(alignment.calculateShift(image_stack[idx['g1'][0]::2, idx['g1'][1]::2, i],
                                                         image_stack[idx['g1'][0]::2, idx['g1'][1]::2,
                                                         ref_frame_idx])) * 2
            # Don't correct big motions
            if np.max(np.abs(shift[n])) > max_global_motion:
                shift[n] = [0, 0]

            extension = image_stack[:, :, i].shape - np.abs(np.flip(shift[n]))
            x_start1 = np.maximum(-shift[n][0], 0)
            y_start1 = np.maximum(-shift[n][1], 0)
            x_start2 = np.maximum(shift[n][0], 0)
            y_start2 = np.maximum(shift[n][1], 0)
            image_stack_aligned[y_start1:y_start1 + extension[0], x_start1:x_start1 + extension[1], i]\
                = image_stack[y_start2:y_start2 + extension[0], x_start2:x_start2 + extension[1], i].copy()
            n += 1

        return (image_stack_aligned.astype(np.float32) / scaling).mean(axis=2).astype(im_type)


def bl_correction(image: np.ndarray, blacklevel: int) -> np.array:
    """
    Black level correction.

    Args:
        image: RAW image.
        blacklevel: Black level.

    Returns:
        RAW image after black level correction.
    """
    im_type = image.dtype

    return np.clip(image.astype(np.float32) - blacklevel, 0, None).astype(im_type)


def white_balance(image: np.ndarray, pattern: str, method: str = 'manual', wb_gains: np.ndarray = None,
                  bitdepth: int = 8, black_level: int = 0, white_level: int = 1023, norm: float = None,
                  sigma: float = 1, grad_order: int = 1) -> np.ndarray:
    """
    Perform white balance on an input RAW image using one of the following methods:
    - manual: Manual white balance with the provided gains.
    - greyworld: Grey world algorithm.
    - greyedge: J. Van de Weijer, T. Gevers and A. Gijsenji, “Edge-Based Color Constancy,” IEEE Transactions on Image
     Processing, vol. 16, no. 9, September 2007.

    Args:
        image: RAW image.
        pattern: Bayer pattern, one of {'rggb, 'grbg', 'gbrg', 'bggr'}.
        method: White balance method, one of {'manual', 'greyworld', 'greyedge'}.
        wb_gains: White balance gains [red, green, blue] for manual white balance.
        bitdepth: RAW image bitdepth.
        black_level: Minimum value of the valid pixels used in the computations.
        white_level: Maximum value of the valid pixels used in the computations.
        norm: Minkowski norm. If None, the average is used instead.
        sigma: (greyedge) Standard deviation of the Gaussian smoothing operation applied.
        grad_order: (greyedge) Derivative order in the range [0, 2].

    Returns:
        White-balanced RAW image.
    """
    max_pix_value = (2 ** bitdepth - 1)
    im_type = image.dtype

    image = image.astype(np.float32)

    # Manual WB
    if method == 'manual':
        if wb_gains is not None:
            for c, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
                if c[0] == 'r':
                    image[y::2, x::2] *= wb_gains[0]
                elif c[0] == 'b':
                    image[y::2, x::2] *= wb_gains[2]
                else:
                    image[y::2, x::2] *= wb_gains[1]
        else:
            raise ValueError('wb_gains need to be supplied when manual WB is used')

    else:
        # Estimation functions
        if norm is None:
            func = np.mean
        elif norm == -1:  # inf
            func = np.max
        else:
            func = lambda x: np.power(np.sum(np.power(x, norm)), 1 / norm)

        if method == 'greyedge':
            image_orig = image.copy()
            image = smooth_gradient(image, sigma, grad_order)

        # Channel average/norm
        avg = dict((c, float(0.0)) for c in 'rgb')
        for c, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
            ch = image[y::2, x::2]
            avg[c] += func(ch[(black_level <= ch) & (ch <= white_level)])
        avg['g'] /= 2  # Channel average

        if method == 'greyedge':
            image = image_orig

        # Discount the illuminant
        for c, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
            if c[0] != 'g':
                image[y::2, x::2] = image[y::2, x::2] * (avg['g'] + EPS) / (avg[c] + EPS)

    return np.clip(np.round(image), 0, max_pix_value).astype(im_type)


def demosaicing(image: np.ndarray, pattern: str, bitdepth: int = 8) -> np.ndarray:
    """
    Simple demosaicing using OpenCV.

    Args:
        image: Input RAW image.
        pattern: Bayer pattern, one of {'rggb, 'grbg', 'gbrg', 'bggr'}.
        bitdepth: RAW image bitdepth.

    Returns:
         RGB image.
    """
    max_pix_value = (2 ** bitdepth - 1)
    im_type = image.dtype

    bayer_pattern_cv = {'bggr': cv2.COLOR_BayerBG2BGR, 'gbrg': cv2.COLOR_BayerGB2BGR,
                        'rggb': cv2.COLOR_BayerRG2BGR, 'grbg': cv2.COLOR_BayerGR2BGR}

    image = np.round(image.astype(np.float32) / max_pix_value * (2 ** 16 - 1)).astype(np.uint16)
    image = cv2.cvtColor(image, bayer_pattern_cv[pattern])

    return np.clip(np.round(image / 65535. * max_pix_value), 0, max_pix_value).astype(im_type)


def colour_correction(image: np.ndarray, ccm: np.ndarray, bitdepth: int = 8) -> np.ndarray:
    """
    Apply Colour Conversion Matrix (CCM) to an image.

    Args:
        image: Input RGB image.
        ccm: Colour Correction Matrix of size 3 x 3.
        bitdepth: Input image bitdepth.

    Returns:
         Image with CCM applied.
    """
    max_pix_value = (2 ** bitdepth - 1)
    im_type = image.dtype

    image = image.astype(np.float32) / max_pix_value  # Range [0, 1]
    image_col = image.reshape(-1, 3)

    img_out = np.matmul(image_col, ccm.T).reshape(image.shape)

    return np.clip(np.round(img_out * max_pix_value), 0, max_pix_value).astype(im_type)


def gamma_correction(image: np.ndarray, bitdepth: int = 8) -> np.array:
    """
    Apply sRGB gamma.

    Args:
        image: Linear RGB image.
        bitdepth: Input image bitdepth.

    Returns:
        sRGB image.
    """
    max_pix_value = (2 ** bitdepth - 1)
    im_type = image.dtype

    image = image.astype(np.float32) / max_pix_value  # Range [0, 1]
    srgb = np.where(image > 0.0031308, 1.055 * (image ** (1.0 / 2.4)) - 0.055, 12.92 * image)

    return np.clip(np.round(srgb * max_pix_value), 0, max_pix_value).astype(im_type)


def simple_isp(image_stack: np.ndarray, motion_comp: bool, bitdepth: int, pattern: str,
               blacklevel: int, ccm: np.ndarray, wb_method: str, wb_gains: np.ndarray, wb_bl: int = 0,
               wb_wl: int = 1023, wb_norm: float = None, wb_sigma: float = 0, wb_grad_order: int = 0) -> np.ndarray:
    """
    Simplified ISP performing denoising, black-level correction, white balance, demosaicing, colour and gamma
     correction.

    Args:
        image_stack: Stack of images of size H x W x NumImages.
        motion_comp: Toggle to activate motion compensation.
        bitdepth: RAW image bitdepth.
        pattern: Bayer pattern, one of {'rggb, 'grbg', 'gbrg', 'bggr'}.
        blacklevel: Black level.
        ccm: Colour Correction Matrix of size 3 x 3.
        wb_method: White balance method, one of {'manual', 'greyworld', 'greyedge'}.
        wb_gains: White balance gains [red, green, blue] for manual white balance.
        wb_bl: Minimum value of the valid pixels used in the computations.
        wb_wl: Maximum value of the valid pixels used in the computations.
        wb_norm: Minkowski norm. If None, the average is used instead.
        wb_sigma: (greyedge) Standard deviation of the Gaussian smoothing operation applied.
        wb_grad_order: (greyedge) Derivative order in the range [0, 2].

    Returns:
        Processed 8-bit RGB image.
    """
    max_pix_value = (2 ** bitdepth - 1)

    output = denoising(image_stack, pattern, motion_comp, bitdepth)
    output = bl_correction(output, blacklevel)
    output = white_balance(output, pattern, wb_method, wb_gains, bitdepth, wb_bl, wb_wl, wb_norm, wb_sigma,
                           wb_grad_order)
    output = demosaicing(output, pattern, bitdepth)
    output = colour_correction(output, ccm, bitdepth)
    output = gamma_correction(output, bitdepth)

    return np.clip(np.round(output.astype(np.float32) / max_pix_value * 255), 0, 255).astype(np.uint8)
