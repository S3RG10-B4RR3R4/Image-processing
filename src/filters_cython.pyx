import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# Definir tipos
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.uint8_t DTYPE_uint8_t

def gaussian_filter_cython(np.ndarray[DTYPE_t, ndim=2] imagen):
    """
    Implementación de filtro Gaussiano en Cython.
    
    Args:
        imagen (numpy.ndarray): Imagen de entrada en escala de grises
    
    Returns:
        numpy.ndarray: Imagen suavizada
    """
    # Kernel Gaussiano 3x3
    cdef np.ndarray[DTYPE_t, ndim=2] kernel = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 4.0, 2.0],
        [1.0, 2.0, 1.0]
    ]) / 16.0

    cdef int altura = imagen.shape[0]
    cdef int ancho = imagen.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] resultado = np.zeros((altura, ancho), dtype=DTYPE)

    cdef int i, j
    cdef double suma

    for i in range(1, altura-1):
        for j in range(1, ancho-1):
            suma = (imagen[i-1, j-1] * kernel[0, 0] +
                    imagen[i-1, j] * kernel[0, 1] +
                    imagen[i-1, j+1] * kernel[0, 2] +
                    imagen[i, j-1] * kernel[1, 0] +
                    imagen[i, j] * kernel[1, 1] +
                    imagen[i, j+1] * kernel[1, 2] +
                    imagen[i+1, j-1] * kernel[2, 0] +
                    imagen[i+1, j] * kernel[2, 1] +
                    imagen[i+1, j+1] * kernel[2, 2])
            resultado[i, j] = suma

    return resultado.astype(np.uint8)

def sobel_filter_cython(np.ndarray[DTYPE_t, ndim=2] imagen):
    """
    Implementación de filtro Sobel en Cython.
    
    Args:
        imagen (numpy.ndarray): Imagen de entrada en escala de grises
    
    Returns:
        numpy.ndarray: Imagen con bordes detectados
    """
    # Kernels de Sobel
    cdef np.ndarray[DTYPE_t, ndim=2] kernel_x = np.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ])

    cdef np.ndarray[DTYPE_t, ndim=2] kernel_y = np.array([
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0]
    ])

    cdef int altura = imagen.shape[0]
    cdef int ancho = imagen.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] resultado = np.zeros((altura, ancho), dtype=DTYPE)

    cdef int i, j
    cdef double gx, gy, g

    for i in range(1, altura-1):
        for j in range(1, ancho-1):
            # Gradiente en X
            gx = (imagen[i-1, j-1] * kernel_x[0, 0] +
                  imagen[i-1, j] * kernel_x[0, 1] +
                  imagen[i-1, j+1] * kernel_x[0, 2] +
                  imagen[i, j-1] * kernel_x[1, 0] +
                  imagen[i, j] * kernel_x[1, 1] +
                  imagen[i, j+1] * kernel_x[1, 2] +
                  imagen[i+1, j-1] * kernel_x[2, 0] +
                  imagen[i+1, j] * kernel_x[2, 1] +
                  imagen[i+1, j+1] * kernel_x[2, 2])

            # Gradiente en Y
            gy = (imagen[i-1, j-1] * kernel_y[0, 0] +
                  imagen[i-1, j] * kernel_y[0, 1] +
                  imagen[i-1, j+1] * kernel_y[0, 2] +
                  imagen[i, j-1] * kernel_y[1, 0] +
                  imagen[i, j] * kernel_y[1, 1] +
                  imagen[i, j+1] * kernel_y[1, 2] +
                  imagen[i+1, j-1] * kernel_y[2, 0] +
                  imagen[i+1, j] * kernel_y[2, 1] +
                  imagen[i+1, j+1] * kernel_y[2, 2])

            # Magnitud del gradiente
            g = sqrt(gx*gx + gy*gy)
            resultado[i, j] = min(255.0, max(0.0, g))

    return resultado.astype(np.uint8)

def median_filter_cython(np.ndarray[DTYPE_t, ndim=2] imagen):
    """
    Implementación de filtro de Mediana en Cython.
    
    Args:
        imagen (numpy.ndarray): Imagen de entrada en escala de grises
    
    Returns:
        numpy.ndarray: Imagen con ruido reducido
    """
    cdef int altura = imagen.shape[0]
    cdef int ancho = imagen.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] resultado = np.zeros((altura, ancho), dtype=DTYPE)

    cdef int i, j, k
    cdef double[9] vecindad

    for i in range(1, altura-1):
        for j in range(1, ancho-1):
            # Extraer vecindad 3x3
            k = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    vecindad[k] = imagen[i+di, j+dj]
                    k += 1

            # Ordenar manualmente (bubble sort simple)
            for a in range(9):
                for b in range(a+1, 9):
                    if vecindad[a] > vecindad[b]:
                        vecindad[a], vecindad[b] = vecindad[b], vecindad[a]

            # La mediana es el elemento central (posición 4)
            resultado[i, j] = vecindad[4]

    return resultado.astype(np.uint8)
