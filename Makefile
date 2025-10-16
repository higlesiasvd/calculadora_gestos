IMAGE := gest-calc

.PHONY: build camtest run shell clean

build:
	docker build -t $(IMAGE) .

camtest:
	docker run -it \
		--shm-size=24g \
		-e DISPLAY=$$DISPLAY \
		-e QT_QPA_PLATFORM_PLUGIN_PATH=/opt/.venv/lib/python3.9/site-packages/cv2/qt/plugins \
		-e QT_X11_NO_MITSHM=1 \
		-v ./src:/opt/project \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /dev/:/dev/ \
		--privileged \
		--rm \
		$(IMAGE) python /opt/project/webcam_test.py

run:
	docker run -it \
		--shm-size=24g \
		-e DISPLAY=$$DISPLAY \
		-e QT_QPA_PLATFORM_PLUGIN_PATH=/opt/.venv/lib/python3.9/site-packages/cv2/qt/plugins \
		-e QT_X11_NO_MITSHM=1 \
		-v ./src:/opt/project \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /dev/:/dev/ \
		--privileged \
		--rm \
		$(IMAGE) python /opt/project/main.py

shell:
	docker run -it \
		--shm-size=24g \
		-e DISPLAY=$$DISPLAY \
		-e QT_QPA_PLATFORM_PLUGIN_PATH=/opt/.venv/lib/python3.9/site-packages/cv2/qt/plugins \
		-e QT_X11_NO_MITSHM=1 \
		-v ./src:/opt/project \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /dev/:/dev/ \
		--privileged \
		--rm \
		$(IMAGE) /bin/bash

clean:
	docker rmi $(IMAGE)

help:
	@echo "Comandos disponibles:"
	@echo "  make build    - Construir la imagen Docker"
	@echo "  make camtest  - Probar la cámara"
	@echo "  make run      - Ejecutar la calculadora gestual"
	@echo "  make shell    - Abrir terminal en el contenedor"
	@echo "  make clean    - Eliminar la imagen Docker"