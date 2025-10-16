# Variables
VENV := ../../.venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
SRC_DIR := src

.PHONY: install camtest run test clean help

# Instalar dependencias en el entorno virtual
install:
	@echo "Instalando dependencias..."
	$(PIP) install -r requirements.txt
	@echo "Dependencias instaladas correctamente"

# Probar la cámara con webcam_test.py
camtest:
	@echo "Probando cámara..."
	cd $(SRC_DIR) && $(PYTHON) webcam_test.py

# Ejecutar la calculadora gestual
run:
	@echo "Iniciando calculadora gestual..."
	cd $(SRC_DIR) && $(PYTHON) main.py

# Ejecutar tests (si existen)
test:
	@echo "Ejecutando tests..."
	cd $(SRC_DIR) && $(PYTHON) -m pytest -v 2>/dev/null || echo "No hay tests configurados"

# Limpiar archivos temporales y cache
clean:
	@echo "Limpiando archivos temporales..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Limpieza completada"

# Mostrar ayuda
help:
	@echo "Comandos disponibles:"
	@echo ""
	@echo "  make install  - Instalar dependencias en el entorno virtual"
	@echo "  make camtest  - Probar la cámara (webcam_test.py)"
	@echo "  make run      - Ejecutar la calculadora gestual"
	@echo "  make test     - Ejecutar tests unitarios"
	@echo "  make clean    - Limpiar archivos temporales y cache"
	@echo "  make help     - Mostrar esta ayuda"
	@echo ""
	@echo "Estructura modular:"
	@echo "  src/config/         - Configuración de accesibilidad"
	@echo "  src/voice/          - Sistema de voz"
	@echo "  src/core/           - Calculadora y detección de gestos"
	@echo "  src/ui/             - Interfaz gráfica"
	@echo "  src/app/            - Aplicación principal"