import numpy as np
import cv2

def gaussian_filter_numpy(image):
    """
    Optimized Gaussian filter implementation using NumPy.
    
    Args:
        image (numpy.ndarray): Grayscale input image
    
    Returns:
        numpy.ndarray: Smoothed image
    """
    # 3x3 Gaussian kernel
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16.0

    result = cv2.filter2D(image, -1, kernel)
    return result

def sobel_filter_numpy(image):
    """
    Optimized Sobel filter implementation using NumPy.
    
    Args:
        image (numpy.ndarray): Grayscale input image
    
    Returns:
        numpy.ndarray: Edge-detected image
    """
    # Sobel kernels
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    # Apply convolution with NumPy
    gx = cv2.filter2D(image.astype(float), -1, kernel_x)
    gy = cv2.filter2D(image.astype(float), -1, kernel_y)

    # Compute gradient magnitude
    g = np.sqrt(gx**2 + gy**2)
    return np.clip(g, 0, 255).astype(np.uint8)

def median_filter_numpy(image):
    """
    Optimized Median filter implementation using NumPy.
    
    Args:
        image (numpy.ndarray): Grayscale input image
    
    Returns:
        numpy.ndarray: Noise-reduced image
    """
    return cv2.medianBlur(image, 3)
