# can be changed, however this offers a good compromise between recency and compatibility
# Imagen base con Python 3.9
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV QT_QPA_PLATFORM_PLUGIN_PATH=$VIRTUAL_ENV/lib/python3.9/site-packages/cv2/qt/plugins
ENV QT_X11_NO_MITSHM=1

# Dependencias del sistema para OpenCV, MediaPipe y GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libqt5gui5 \
        libqt5widgets5 \
        libqt5core5a \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear entorno virtual e instalar dependencias de Python
RUN python -m venv /opt/.venv \
    && /opt/.venv/bin/pip install --upgrade pip \
    && /opt/.venv/bin/pip install --no-cache-dir \
        opencv-python==4.8.1.78 \
        mediapipe==0.10.8 \
        numpy==1.24.3

WORKDIR /opt

CMD ["/bin/bash"]