import numpy as np
import cv2

def ampliarRangoDin(imagen):
    imagen = np.float32(imagen);
    imagenPlana = imagen.reshape(1, -1)
    pixelesOrd = np.sort(imagenPlana)
    al = np.unique(pixelesOrd)[5]
    ah = np.unique(pixelesOrd)[-6]
    imagenAmpliada = (imagen - al) * (255 /(ah - al))
    np.clip(imagenAmpliada, 0, 255)
    imagenAmpliada = np.uint8(np.clip(imagenAmpliada, 0, 255))
    return imagenAmpliada

from scipy.ndimage import median_filter
def filtroMediana (imagen, n = None):
    imagen = np.float32(imagen)
    if (n is None):
        n = 3
    ndiv = n //2
    imagen_filtrada = np.zeros_like(imagen)
    imagen_ampliada = cv2.copyMakeBorder(imagen, ndiv, ndiv, ndiv, ndiv, cv2.BORDER_REPLICATE)
    ventana = n * 2 + 1
    imagen_filtrada = median_filter(imagen, size=ventana)

    imagen_filtrada = np.uint8(imagen_filtrada)
    return imagen_filtrada


def redimensionar(imagen):
    imagenRed = cv2.resize(imagen, (128, 128))
    return imagenRed

def filtro(imagen, mascara):
    imagen = np.float32(imagen)
    imagen_f = cv2.filter2D(imagen, -1, mascara)
    return imagen_f

def gradiente(imagen):
    m_g_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]] )
    m_g_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ix = filtro(imagen, m_g_x)
    Iy = filtro(imagen, m_g_y)
    E = np.sqrt(Ix**2 + Iy**2)
    Phi = np.arctan2(Ix, Iy)
    return E

def umbralizacion(imagen):
    q = np.median(imagen)
    segmentada = imagen
    segmentada[imagen>q] = 255
    return segmentada

# Extracción de características HOG
def extraccion_hog(imagen):
    fd, imagen_hog = hog(
        imagen,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True  
    )
    return fd, imagen_hog