#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[ ]:


"""Ideas:
- ampliar rango dinamico 
- Podriamos hacer homogenizar el histograma
- Filtro de suavizado 
- Filtro de sal y pimienta(sobre todo sal y pimienta)


# In[ ]:


como subir documentos
git pull
modificar
git add . ( o nombre del archivo)
git commit -m "descripcion de los cambios realizados"

"""
# In[4]:

"""
imagen = cv2.imread('cat_dog_100/train/dog/dog.10002.jpg', 0)
print(np.min(imagen), np.max(imagen));
imagenAmpliada = ampliarRangoDin(imagen)
cv2.imshow('imagen original', imagen)
cv2.imshow('imagen ampliada', imagenAmpliada)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
# In[3]:


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


# In[18]:


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
    


# In[3]:


from skimage.util import random_noise

imagen = cv2.imread('cat_dog_100/train/dog/dog.10095.jpg', 0)
imagen_sp = np.uint8(255 * random_noise(imagen, mode= 's&p', amount = 0.1))
imagenFiltradaMedia = (filtroMediana(imagen_sp))
cv2.imshow('imagen original', imagen_sp)
cv2.imshow('imagen filtrada', imagenFiltradaMedia)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[12]:


#Redimensionar imagenes
imagen1 = cv2.imread('cat_dog_100/train/dog/dog.10095.jpg', 0)
imagen2 = cv2.imread('cat_dog_100/train/dog/dog.10005.jpg', 0)
imagen3 = cv2.imread('cat_dog_100/train/dog/dog.10015.jpg', 0)
cv2.imshow('imagen 1', imagen1)
cv2.imshow('imagen 2', imagen2)
cv2.imshow('imagen 3', imagen3)
cv2.waitKey(0)
cv2.destroyAllWindows()
imagenRed1 = cv2.resize(imagen1, (128, 128))
imagenRed2 = cv2.resize(imagen2, (128, 128))
imagenRed3 = cv2.resize(imagen3, (128, 128))
cv2.imshow('imagen 1', imagenRed1)
cv2.imshow('imagen 2', imagenRed2)
cv2.imshow('imagen 3', imagenRed3)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


def redimensionar(imagen):
    imagenRed = cv2.resize(imagen, (128, 128))
    return imagenRed


# In[19]:


for i in range(15):
    camino = 'cat_dog_100/train/dog/dog.' + str(10000 + i + 1) + '.jpg'
    imagen1 = cv2.imread(camino, 0)
    # Muestra la imagen en una ventana con un nombre único
    cv2.imshow(f'imagen_{i}', imagen1)  # f'imagen_{i}' genera nombres como 'imagen_0', 'imagen_1', etc.

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[25]:


print(np.mean([3, 5]))


# In[ ]:


#Bordes 


# In[36]:


def filtro(imagen, mascara):
    imagen = np.float32(imagen)
    imagen_f = cv2.filter2D(imagen, -1, mascara)
    return imagen_f


# In[37]:


def gradiente(imagen):
    m_g_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]] )
    m_g_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ix = filtro(imagen, m_g_x)
    Iy = filtro(imagen, m_g_y)
    E = np.sqrt(Ix**2 + Iy**2)
    Phi = np.arctan2(Ix, Iy)
    return E, Phi


# In[ ]:





# In[38]:


imagen = cv2.imread('cat_dog_100/train/dog/dog.10095.jpg', 0)
E, phi = gradiente(imagen)

cv2.imshow('Magnitudes',E/E.max())
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[29]:


import numpy as np
import cv2 as cv

filename = 'cat_dog_100/train/dog/dog.10009.jpg'
img = cv.imread(filename)
img = redimensionar(img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
gray = filtroMediana(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]= 255

cv.imshow('dst',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#Probar a umbralizar, como en la imagen del platano, sino hay mucha mierda


# In[11]:


def imagen_media_chat(imagen, etiquetas, centros):
    print(centros)
    etiquetas = etiquetas.reshape(imagen.shape)
    imagen_segmentada = np.zeros_like(imagen, dtype=np.float32)    
    for i, centro in enumerate(centros):
        imagen_segmentada[etiquetas == i] = centro
    
    return imagen_segmentada.astype(np.uint8)  # Convertir a uint8 para visualización


# In[12]:


from sklearn.cluster import KMeans
imagen = cv2.imread('cat_dog_100/train/dog/dog.10009.jpg', 0)
kmeans = KMeans(n_clusters = 5, n_init=10)
kmeans.fit(imagen.reshape(-1, 1))
centros = kmeans.cluster_centers_
etiquetas = kmeans.labels_
imagenMedia = imagen_media_chat(imagen, etiquetas, centros)
cv2.imshow('Imagen media', imagenMedia)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[17]:


def calcular_caracteristicas_hog(imagen, tamano_celda=16, num_bins=9):
    # nos aseguramos de que la imagen esté en escala de grises
    if len(imagen.shape) > 2:
        raise ValueError("La imagen de entrada debe estar en escala de grises")

    # Paso 1: Calculamos el gradiente usando filtros [-1, 0, 1]
    grad_x = cv2.filter2D(imagen, cv2.CV_32F, np.array([[-1, 0, 1]]))
    grad_y = cv2.filter2D(imagen, cv2.CV_32F, np.array([[-1], [0], [1]]))

    # Paso 2: Calculamos la magnitud y orientación del gradiente
    magnitud = np.sqrt(grad_x**2 + grad_y**2)
    orientacion = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 180

    # Paso 3: Dividimos la imagen en celdas
    h, w = imagen.shape
    celdas_x = w // tamano_celda
    celdas_y = h // tamano_celda

    caracteristicas_hog = []

    # Paso 4: Calculamos histogramas para cada celda
    for i in range(celdas_y):
        for j in range(celdas_x):
            # Extraemos la región de la celda
            celda_mag = magnitud[i * tamano_celda:(i + 1) * tamano_celda, j * tamano_celda:(j + 1) * tamano_celda]
            celda_ori = orientacion[i * tamano_celda:(i + 1) * tamano_celda, j * tamano_celda:(j + 1) * tamano_celda]

            # Creamos el histograma para la celda
            histograma = np.zeros(num_bins, dtype=np.float32)
            ancho_bin = 180 / num_bins

            for m in range(celda_mag.shape[0]):
                for n in range(celda_mag.shape[1]):
                    # Determinar el bin para cada píxel
                    indice_bin = int(celda_ori[m, n] // ancho_bin) % num_bins
                    histograma[indice_bin] += celda_mag[m, n]

            # Normalizamos el histograma (norma L2)
            norma = np.linalg.norm(histograma)
            if norma != 0:
                histograma /= norma

            caracteristicas_hog.append(histograma)

    # Paso 5: Concatenamos todos los histogramas para formar el vector de características
    caracteristicas_hog = np.concatenate(caracteristicas_hog)

    return caracteristicas_hog



# In[18]:


def pipeline_completo(imagen_path):
    # Cargar imagen en escala de grises
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    imagen = cv2.resize(imagen, (128, 128))  # Redimensionar para consistencia
    
    # 1. Preprocesamiento
    imagen_ampliada = ampliarRangoDin(imagen)
    imagen_suavizada = filtroMediana(imagen_ampliada)
    
    # 2. Extracción de características HOG
    caracteristicas_hog = calcular_caracteristicas_hog(imagen_suavizada, tamano_celda=16, num_bins=9)
    
    # 3. Puntos clave (Harris)
    gray = np.float32(imagen_suavizada)
    puntos_harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    puntos_harris = puntos_harris.flatten()[:500]  # Limitar número de puntos clave
    
    # 4. Segmentación por K-means
    kmeans = KMeans(n_clusters=5, n_init=10)
    kmeans.fit(imagen_suavizada.reshape(-1, 1))
    centros = kmeans.cluster_centers_
    etiquetas = kmeans.labels_
    imagen_segmentada = imagen_media_chat(imagen_suavizada, etiquetas, centros).flatten()
    
    # 5. Combinar características
    caracteristicas_combinadas = np.hstack([caracteristicas_hog, puntos_harris, imagen_segmentada])
    
    return caracteristicas_combinadas

# Ejemplo de uso
imagen_path = 'cat_dog_100/train/dog/dog.10095.jpg'
caracteristicas = pipeline_completo(imagen_path)
print(f"Vector de características combinado: {caracteristicas.shape}")


# In[2]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


X_train, X_test, y_train, y_test = train_test_split(caracteristicas, etiquetas, test_size=0.15, random_state=42)

# Crear y entrenar el modelo
modelo = SVC(kernel='linear', random_state=42)
modelo.fit(X_train, y_train)

# Evaluar el modelo
predicciones = modelo.predict(X_test)
print(f"Precisión del modelo: {accuracy_score(y_test, predicciones):.2f}")

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar
predicciones = modelo.predict(X_test)
print(f"Precisión del modelo: {accuracy_score(y_test, predicciones):.2f}")

print(confusion_matrix(y_test, predicciones))
print(classification_report(y_test, predicciones))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

nueva_imagen = cv2.imread('cat_dog_100/train/dog/dog.10095.jpg', cv2.IMREAD_GRAYSCALE)
caracteristicas_nueva = calcular_caracteristicas_comb(nueva_imagen)
prediccion = modelo.predict([caracteristicas_nueva])

print("Es un perro" if prediccion[0] == 1 else "Es un gato")


# In[3]:


import numpy as np
import cv2 as cv

img = cv.imread('cat_dog_100/train/dog/dog.10095.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray,None)

img=cv.drawKeypoints(gray,kp,img)

          
cv.imshow('dst',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

