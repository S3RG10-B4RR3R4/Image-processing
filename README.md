# Image Processing Project

Welcome to the **Image Processing Project**! This project explores different image processing techniques using **Python**, **NumPy**, and **Cython**. We implement and compare the performance of three image filters: **Gaussian**, **Sobel**, and **Median**. The goal is to highlight the performance improvements achieved using NumPy and Cython over pure Python implementations.

## 📦 Installation

To get started, clone the repository and install the required dependencies by running the following command:

```bash
git clone https://github.com/yourusername/Image-processing.git
cd Image-processing
pip install -r requirements.txt
```

🔧 Additional Setup for Cython
After installing the dependencies, you will need to install the Microsoft C++ Build Tools to compile the Cython code on Windows. Follow these steps:

Install Microsoft C++ Build Tools:
Download and install the tools from the following link: Microsoft Visual C++ Build Tools.
During installation, ensure that you select C++ build tools under the Workloads section.
🚨 Cython Compilation (Windows)
To compile the Cython modules on Windows, you'll need to have the Visual Studio Build Tools installed. If you already have Microsoft Visual C++ Build Tools installed, follow these steps:

Run the command to set up the Visual Studio environment:

Open Git Bash or CMD and run the following command to set up the Visual Studio build tools in your terminal:

bash
Copiar
Editar
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
This will configure the environment to use the build tools in the terminal.

Navigate to the project folder:

Change the directory to the folder where the setup.py file and Cython code are located:

bash
Copiar
Editar
cd /c/Users/venta/Downloads/PRUEBA-USO/Image-processing/src
Compile the Cython code:

Finally, run the following command to compile the Cython file (filters_cython.pyx):

bash
Copiar
Editar
python setup.py build_ext --inplace
This step will generate the .pyd file (or .so on other systems) that you can use in your Python project.

This version of the README.md includes the Cython compilation setup instructions in English and provides clear steps for users working on Windows.
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
