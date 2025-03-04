import numpy as np

def gaussian_filter_python(image):
    """
    Gaussian filter implementation in pure Python.
    
    Args:
        image (numpy.ndarray): Grayscale input image
    
    Returns:
        numpy.ndarray: Smoothed image
    """
    height, width = image.shape
    result = np.zeros_like(image, dtype=float)

    # 3x3 Gaussian kernel
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16.0

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract 3x3 neighborhood
            neighborhood = image[i-1:i+2, j-1:j+2]
            # Apply kernel
            result[i, j] = np.sum(neighborhood * kernel)

    return result.astype(np.uint8)

def sobel_filter_python(image):
    """
    Sobel filter implementation in pure Python.
    
    Args:
        image (numpy.ndarray): Grayscale input image
    
    Returns:
        numpy.ndarray: Edge-detected image
    """
    height, width = image.shape
    result = np.zeros_like(image, dtype=float)

    # Sobel kernels for X and Y directions
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract 3x3 neighborhood
            neighborhood = image[i-1:i+2, j-1:j+2]

            # Apply kernels
            gx = np.sum(neighborhood * kernel_x)
            gy = np.sum(neighborhood * kernel_y)

            # Compute gradient magnitude
            g = np.sqrt(gx**2 + gy**2)
            result[i, j] = np.clip(g, 0, 255)

    return result.astype(np.uint8)

def median_filter_python(image):
    """
    Median filter implementation in pure Python.
    
    Args:
        image (numpy.ndarray): Grayscale input image
    
    Returns:
        numpy.ndarray: Noise-reduced image
    """
    height, width = image.shape
    result = np.zeros_like(image)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract 3x3 neighborhood
            neighborhood = image[i-1:i+2, j-1:j+2].flatten()
            # Compute median
            result[i, j] = np.median(neighborhood)

    return result.astype(np.uint8)
