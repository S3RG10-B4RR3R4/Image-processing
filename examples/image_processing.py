import numpy as np 
import matplotlib.pyplot as plt
import time
from PIL import Image

from src.filters_python import (
    gaussian_filter_python, 
    sobel_filter_python, 
    median_filter_python
)
from src.filters_numpy import (
    gaussian_filter_numpy,
    sobel_filter_numpy,
    median_filter_numpy
)
from src.filters_cython import (
    gaussian_filter_cython,
    sobel_filter_cython,
    median_filter_cython
)

def run_comprehensive_tests(image):
    """
    Runs performance tests for different filter implementations.
    
    Args:
        image (numpy.ndarray): Input grayscale image
    
    Returns:
        tuple: Dictionaries of results and execution times
    """
    # Convert the image to float64 for Cython compatibility
    image_float = image.astype(np.float64)

    results = {}
    times = {}

    # Dictionary of filter functions
    filters = {
        'gaussian': [
            (gaussian_filter_python, 'Python'),
            (gaussian_filter_numpy, 'NumPy'),
            (gaussian_filter_cython, 'Cython')
        ],
        'sobel': [
            (sobel_filter_python, 'Python'),
            (sobel_filter_numpy, 'NumPy'),
            (sobel_filter_cython, 'Cython')
        ],
        'median': [
            (median_filter_python, 'Python'),
            (median_filter_numpy, 'NumPy'),
            (median_filter_cython, 'Cython')
        ]
    }

    # Apply each filter and measure execution times
    for filter_type, implementations in filters.items():
        for filter_func, implementation in implementations:
            start = time.time()
            key = f'{filter_type}_{implementation.lower()}'
            
            # Cython requires the image in float64
            if implementation == 'Cython':
                results[key] = filter_func(image_float)
            else:
                results[key] = filter_func(image)
            
            times[key] = time.time() - start

    return results, times

def visualize_results(image, results, times):
    """
    Visualizes the results of different filters.
    
    Args:
        image (numpy.ndarray): Original image
        results (dict): Filter results
        times (dict): Execution times
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    filters = ['gaussian', 'sobel', 'median']
    implementations = ['python', 'numpy', 'cython']

    for i, filter_name in enumerate(filters):
        for j, implementation in enumerate(implementations):
            key = f'{filter_name}_{implementation}'
            axes[i, j+1].imshow(results[key], cmap='gray')
            axes[i, j+1].set_title(f'{filter_name.capitalize()} {implementation.capitalize()}: {times[key]:.4f}s')
            axes[i, j+1].axis('off')

    plt.tight_layout()
    plt.show()

def analyze_performance(times):
    """
    Analyzes and presents the execution times of the filters.
    
    Args:
        times (dict): Filter execution times
    """
    print("Image Filter Performance Analysis")
    print("=" * 50)

    for filter_name in ['gaussian', 'sobel', 'median']:
        print(f"\n{filter_name.capitalize()} Filter:")
        python_time = times[f'{filter_name}_python']
        numpy_time = times[f'{filter_name}_numpy']
        cython_time = times[f'{filter_name}_cython']

        print(f"  Python: {python_time:.4f} seconds")
        print(f"  NumPy Speedup: {python_time/numpy_time:.2f}x")
        print(f"  Cython Speedup: {python_time/cython_time:.2f}x")

def main():
    """
    Main function to demonstrate image processing.
    """
    # Load example image (replace with your own image)
    example_image = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    
    # Run tests
    results, times = run_comprehensive_tests(example_image)
    
    # Visualize results
    visualize_results(example_image, results, times)
    
    # Analyze performance
    analyze_performance(times)

if __name__ == "__main__":
    main()
