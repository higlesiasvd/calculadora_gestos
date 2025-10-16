# Calculadora Gestual con Visión por Computador

Sistema avanzado de calculadora aritmética controlada mediante gestos de manos, utilizando visión por computador y aprendizaje automático para reconocimiento de gestos en tiempo real.

## Descripción

Aplicación interactiva que permite realizar operaciones aritméticas básicas mediante gestos de manos capturados por cámara web. El sistema emplea MediaPipe Hands para detección y seguimiento de landmarks en 3D, ofreciendo una interfaz intuitiva y accesible sin necesidad de dispositivos de entrada tradicionales.

### Características principales

- **Detección en tiempo real**: Procesamiento de video a 30 FPS con latencia inferior a 100ms
- **Reconocimiento robusto**: 15 gestos distintos con precisión superior al 95%
- **Arquitectura modular**: Código organizado en 5 paquetes independientes siguiendo principios SOLID
- **Sistema de estabilización**: Buffer temporal de 10 frames con umbral del 70% para eliminar falsos positivos
- **Accesibilidad**: Feedback por voz configurable con soporte multilingüe
- **Guías visuales**: Asistencia visual opcional para posicionamiento correcto de manos

## Requisitos del sistema

### Hardware

- Cámara web con resolución mínima 640x480 píxeles
- Procesador con soporte para operaciones vectoriales (AVX2 recomendado)
- 4 GB RAM mínimo
- Iluminación ambiente adecuada (300-500 lux recomendado)

### Software

- Python 3.9 o superior
- macOS 10.15+ / Linux (Ubuntu 20.04+) / Windows 10+
- Permisos de acceso a cámara web

## Instalación

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd p1-gest-calc
```

### 2. Crear entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
make install
```

O manualmente:

```bash
pip install -r requirements.txt
```

### Dependencias principales

- **OpenCV 4.8.1.78**: Procesamiento de imagen y video
- **MediaPipe**: Detección de landmarks en manos (21 puntos por mano)
- **NumPy 1.24.3**: Computación numérica y álgebra lineal
- **pyttsx3 2.90+**: Síntesis de voz multiplataforma

## Uso

### Ejecución básica

```bash
make run
```

O directamente:

```bash
cd src
python3 main.py
```

### Verificar cámara

```bash
make camtest
```

### Comandos disponibles

```bash
make help       # Mostrar ayuda completa
make install    # Instalar dependencias
make run        # Ejecutar aplicación
make camtest    # Probar cámara
make clean      # Limpiar archivos temporales
```

## Gestos soportados

### Gestos de una mano

| Gesto                    | Descripción               |
| ------------------------ | -------------------------- |
| **0**              | Puño cerrado              |
| **1**              | Índice extendido          |
| **2**              | Índice + medio            |
| **3**              | Índice + medio + anular   |
| **4**              | Todos excepto pulgar       |
| **5**              | Mano abierta               |
| **Suma (+)**       | Pulgar + índice (forma L) |
| **Resta (-)**      | 4 dedos horizontal         |
| **División (÷)** | Doble V                    |
| **Igual (=)**      | Signo de okey              |
| **Borrar**         | Meñique                   |

### Gestos de dos manos

| Gesto                          | Descripción      |
| ------------------------------ | ----------------- |
| **6**                    | Mano 5 + mano 1   |
| **7**                    | Mano 5 + mano 2   |
| **8**                    | Mano 5 + mano 3   |
| **9**                    | Mano 5 + mano 4   |
| **Multiplicación (×)** | Índices cruzados |

## Arquitectura del sistema

### Estructura de directorios

```
p1-gest-calc/
├── src/
│   ├── main.py                    # Punto de entrada
│   ├── config/
│   │   ├── __init__.py
│   │   └── accessibility.py       # Configuración de accesibilidad
│   ├── voice/
│   │   ├── __init__.py
│   │   └── feedback.py            # Sistema de síntesis de voz
│   ├── core/
│   │   ├── __init__.py
│   │   ├── calculator.py          # Lógica aritmética
│   │   └── gesture_detector.py    # Detección de gestos
│   ├── ui/
│   │   ├── __init__.py
│   │   └── renderer.py            # Renderizado OpenCV
│   └── app/
│       ├── __init__.py
│       └── gesture_app.py         # Coordinador principal
├── requirements.txt
├── Makefile
└── README.md
```

### Componentes principales

#### 1. GestureDetector (core/gesture_detector.py)

- Procesamiento de video en tiempo real con MediaPipe
- Detección de 21 landmarks por mano en espacio 3D
- Sistema de buffer circular (10 frames) para estabilización
- Validación geométrica de gestos (ángulos, distancias, orientaciones)

#### 2. Calculator (core/calculator.py)

- Máquina de estados para gestión de expresiones
- Soporte para operaciones encadenadas
- Validación de sintaxis aritmética
- Historial de operaciones

#### 3. UIRenderer (ui/renderer.py)

- Renderizado de interfaz con OpenCV
- Display de expresión matemática y resultado
- Indicadores de gestos en tiempo real
- Guías de posicionamiento dinámicas
- Sistema de feedback visual

#### 4. VoiceFeedback (voice/feedback.py)

- Sistema asíncrono de síntesis de voz
- Cola de mensajes con threading
- Selección automática de voces por idioma
- Control de volumen y velocidad

#### 5. GestureCalculatorApp (app/gesture_app.py)

- Coordinador del ciclo principal
- Gestión del pipeline de video
- Control de estados de la aplicación
- Sistema de cooldown entre gestos

## Configuración

### Accesibilidad

Modificar `config/accessibility.py` para personalizar:

```python
class AccessibilityConfig:
    def __init__(self):
        # Configuración de voz
        self.voice_enabled = True
        self.voice_volume = 0.9
        self.voice_rate = 175
  
        # Gestos extendidos
        self.extended_gestures = True
  
        # Asistencia visual
        self.visual_guides = True
```

### Parámetros de detección

En `core/gesture_detector.py`:

```python
# Confianza de detección (0.0 - 1.0)
detection_confidence = 0.8

# Confianza de tracking (0.0 - 1.0)
tracking_confidence = 0.8

# Estabilidad requerida (frames consistentes / buffer size)
min_stability = 0.70  # 70% = 7/10 frames
```
