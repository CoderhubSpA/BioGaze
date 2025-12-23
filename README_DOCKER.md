# BioGaze - Entorno Dockerizado

---

## 🌟 Características principales

- **Python 3.12-slim** como base para compatibilidad y ligereza.
- Soporte para librerías científicas (PyTorch, Ultralytics/YOLO, dlib, OpenCV, mediapipe, pandas, etc.).
- Modelos checkpoint y archivos grandes NO incluidos en la imagen: deben estar en rutas locales bien definidas.
- Carpeta de entrada y salida fáciles de mapear con volúmenes.
- Flujo portable y estable para cualquier usuario.

---

## 📦 Requisitos

- **Docker** y **Docker Compose** 

## descargas e instalacion

sera necesario descargar los siguientes modelos:

**sera necesario crear las carpetas de dlib_checkpoit, input_images, cp**

- Debes descargar el **checkpoint de dlib** desde [este enlace](https://drive.google.com/file/d/1Xkwoou-5xTg_o8zq7hih5bZF3BaXtnU5/view?usp=sharing) y colocarlo en el directorio `dlib_checkpoint` y en el directorio `gaze_estimation/modules/dlib_checkpoint`.
- Debes descargar el **checkpoint del analizador facial (face parser)** desde [este enlace](https://drive.google.com/file/d/1Hvd6tDmCVuF0KUtFIZQ_WXKlQvcHTQbm/view?usp=sharing) y colocarlo en el directorio `face_parser/res/cp`.
- Debes descargar el **checkpoint de estimación de la mirada (gaze estimation)** desde [este enlace](https://drive.google.com/file/d/1bAjhZeTwSgVMf48m8VWfHFBN7EbVjZRW/view?usp=sharing) y colocarlo en el directorio `gaze_estimation/ckpt`.
- Debes descargar el **checkpoint del detector de rostro YOLOv8n-face** (`yolov8n-face.pt`) desde [este enlace](https://github.com/lindevs/yolov8-face?tab=readme-ov-file), se debe descargar la version yolov8x-face.pt (la ultima que sale) y colocarlo en el directorio `detectors/models`. El archivo debe llamarse **exactamente** asi que sera necesario renombrarlo a `yolov8n-face.pt`.

---

---
## **Carpetas**
**sera necesario crear las carpetas de dlib_checkpoit, input_images, cp**

BioGaze/
├── input_images/ # Aquí pones imágenes de entrada (.jpg, .png, etc.)
├── dlib_checkpoint/
│ └── shape_predictor_68_face_landmarks.dat
├── face_parser/
│ └── res/
│ └── cp/
│ └── 79999_iter.pth
├── gaze_estimation/
│ └── ckpt/
│ └── epoch_24_ckpt.pth.tar
│ └── models/
│ └── dlib_checkpoint/
│ └── shape_predictor_68_face_landmarks.dat
├── detectors/
│ └── models/
│ └── yolov8n-face.pt
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── ...otros archivos del repo

## para lanzar el contenedor

usar el comando docker compose up -d --build para crear el contenedor
luego usar el comando docker compose run biogaze bash para entrar en el contenedor
luego usar el comando de python quality_analysis.py -i /app/input_images/image_example.jpg
para que funcionen los script
los resultados saldran en los archivos table_results.txt y verbose_results.txt

`si al intentar levantarlo este falla por no encontrar "verbose_results.txt" intentar entrar otra vez en en contenedor con el comando "docker compose run biogaze bash" y usar hacer el script para que este genere el archi "verbose_results.txt"`

**recordar colocar imagenes dentro de input_images**
