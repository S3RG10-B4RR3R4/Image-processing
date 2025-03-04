# Image Processing Project

Welcome to the **Image Processing Project**! This project explores different image processing techniques using **Python**, **NumPy**, and **Cython**. We implement and compare the performance of three image filters: **Gaussian**, **Sobel**, and **Median**. The goal is to highlight the performance improvements achieved using NumPy and Cython over pure Python implementations.

## 📦 Installation

To get started, clone the repository and install the required dependencies by running the following command:

```bash
git clone https://github.com/yourusername/Image-processing.git
cd Image-processing
pip install -r requirements.txt
```

## 🚀 Usage

After installing the dependencies, you can run the demo script to see the image processing in action:

```bash
python examples/image_processing.py
```

This will load a sample image, apply the filters, and display the results along with execution times for each implementation (Python, NumPy, and Cython).

## 🧑‍💻 Features

* **Filter Implementations**:
   * Gaussian Filter
   * Sobel Filter
   * Median Filter

* **Performance Comparison**:
   * Detailed execution time analysis for Python, NumPy, and Cython implementations.
   * Visual comparison of the filtered images.

* **Image Formats**: The project works with grayscale images and supports fast processing on larger datasets.

## 📈 Performance Analysis

* Measure the time it takes for each filter (Python, NumPy, and Cython) to process an image.
* See the speedup achieved by using **NumPy** and **Cython** over the **pure Python** implementation.

## 📸 Sample Results

### Filtered Results
* **Gaussian Filter**: A smoothing filter that reduces image noise.
* **Sobel Filter**: Edge detection filter.
* **Median Filter**: Reduces noise while preserving edges.

## 📝 License

This project is licensed under the **MIT License**. See the LICENSE file for more details.

## 🤝 Contributing

Feel free to fork this project, submit issues, and contribute to its development! If you find any bugs or have suggestions for improvements, don't hesitate to open an issue.
