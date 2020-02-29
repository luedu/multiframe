import numpy as np
from glob import glob
from scipy.ndimage import gaussian_filter1d


def load_mipi_raw10(filepath: str, height: int, width: int) -> np.array:
    """
    Load MIPI RAW image and convert to numpy format.

    Args:
        filepath: Path to the RAW file.
        height: Image height.
        width: Image width.

    Returns:
        RAW image.
    """

    try:
        with open(filepath, "rb") as f:
            data = f.read()
    except EnvironmentError:
        print('Error when opening RAW file!')

    data = np.frombuffer(data, dtype=np.uint8)

    # 5 bytes contain four 10-bit pixels (5x8 == 4x10)
    b1, b2, b3, b4, b5 = np.reshape(data, (data.shape[0] // 5, 5)).astype(np.uint16).T
    o1 = (b1 << 2) + (b5 & 0x3)
    o2 = (b2 << 2) + ((b5 >> 2) & 0x3)
    o3 = (b3 << 2) + ((b5 >> 4) & 0x3)
    o4 = (b4 << 2) + ((b5 >> 6) & 0x3)
    unpacked = np.reshape(np.concatenate((o1[:, None], o2[:, None], o3[:, None], o4[:, None]), axis=1), 4*o1.shape[0])

    return unpacked.reshape(height, width)


def open_image_stack(folderpath: str, height: int, width: int) -> np.array:
    """
    Open RAW image stack from folder and return it as a numpy array.

    Args:
        folderpath: Path to the RAW stack.
        height: Image height.
        width: Image width.

    Returns:
        Image stack of size H x W x NumImages.
    """

    # Read image names
    stack_list = []
    for filename in glob(folderpath + '*.raw'):
        stack_list.append(filename)
    stack_list.sort()

    # Convert to array
    stack = np.zeros((height, width, len(stack_list)), dtype=np.uint16)
    for i in range(len(stack_list)):
        stack[:, :, i] = load_mipi_raw10(stack_list[i], height, width)

    return stack


def smooth_gradient(image: np.ndarray, sigma: float, order: int) -> np.ndarray:
    """
    Applies gradient of a Gaussian or simply the gradient depending on sigma and order.

    Args:
        image: Input image to be smoothened.
        sigma: Standard deviation of the Gaussian smoothing operation applied.
        order: Derivative order in the range [0, 2]

    Returns:
        Processed image.
    """
    # Apply smoothing and/or the derivative
    if sigma > 0:
        for i in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            img_x = gaussian_filter1d(image[i[0]::2, i[1]::2], sigma=sigma, axis=1, order=order, mode='nearest')
            img_x = gaussian_filter1d(img_x, sigma=sigma, axis=0, order=0, mode='nearest')
            img_y = gaussian_filter1d(image[i[0]::2, i[1]::2], sigma=sigma, axis=0, order=order, mode='nearest')
            img_y = gaussian_filter1d(img_y, sigma=sigma, axis=1, order=0, mode='nearest')
            image[i[0]::2, i[1]::2] = np.sqrt(img_x ** 2 + img_y ** 2)

    elif order > 0:
        for i in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            img_x = np.gradient(image[i[0]::2, i[1]::2], edge_order=order, axis=1)
            img_y = np.gradient(image[i[0]::2, i[1]::2], edge_order=order, axis=0)
            image[i[0]::2, i[1]::2] = np.sqrt(img_x ** 2 + img_y ** 2)

    return image
