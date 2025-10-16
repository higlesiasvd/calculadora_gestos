# Calculadora Gestual

Una calculadora controlada por gestos de manos usando OpenCV y MediaPipe.

## ✅ Código Completado

El código ha sido completado con las siguientes características:

### 🎯 Componentes Principales

1. **GestureDetector** - Detector de gestos con estabilización
   - Detección de manos con MediaPipe
   - Conteo preciso de dedos extendidos
   - Sistema de buffer para estabilización temporal
   - Requiere 7 frames consistentes para confirmar un gesto

2. **Calculator** - Lógica de la calculadora
   - Manejo de números (0-9)
   - Operaciones: suma, resta, multiplicación, división
   - Cálculo de resultados con eval()
   - Funciones de borrado y reset

3. **UIRenderer** - Renderizador de interfaz
   - Pantalla principal con display grande
   - Indicador de gesto actual con barra de confianza
   - Guía de gestos en panel lateral
   - Sistema de feedback visual
   - Barra de cooldown

4. **GestureCalculatorApp** - Aplicación principal
   - Gestión de cámara (1920x1080 @ 30fps)
   - Bucle principal de captura y procesamiento
   - Sistema de cooldown (25 frames)
   - Contador de FPS

### 🖐️ Gestos Reconocidos

#### Números (1 mano):
- **0**: Puño cerrado
- **1**: Solo índice
- **2**: Índice + medio (victoria)
- **3**: Índice + medio + anular
- **4**: Cuatro dedos (sin pulgar)
- **5**: Mano completamente abierta
- **6**: Pulgar + índice
- **7**: Pulgar + índice + medio
- **8**: Pulgar + índice + medio + anular
- **9**: Todos menos meñique

#### Operaciones:
- **Suma (+)**: 1 mano abierta (5 dedos)
- **Resta (-)**: 2 puños cerrados (2 manos)
- **Multiplicar (×)**: 2 manos abiertas
- **Dividir (÷)**: Mano horizontal

#### Control (1 mano):
- **Calcular (=)**: Pulgar arriba
- **Borrar dígito**: Solo meñique levantado
- **Borrar todo**: Pulgar hacia abajo

### 🔧 Correcciones Aplicadas

1. ✅ Todas las fuentes `FONT_HERSHEY_BOLD` reemplazadas por `FONT_HERSHEY_DUPLEX`
2. ✅ Todos los emoticonos eliminados (✓, ➕, ➖, ✖️, ➗, 🧮)
3. ✅ Código completado (métodos `process_gesture` y `run`)
4. ✅ Sistema de estabilización implementado
5. ✅ Interfaz gráfica completa con todos los paneles

### 🚀 Cómo Ejecutar

```bash
cd src
python3 main.py
```

### ⌨️ Controles

- **ESC** o **'q'**: Salir de la aplicación

### 📊 Características Técnicas

- **Resolución**: 1920x1080
- **FPS objetivo**: 30 fps
- **Detección**: MediaPipe Hands (model_complexity=1)
- **Confianza mínima**: 85% detección y tracking
- **Frames para estabilización**: 7 frames consistentes
- **Cooldown**: 25 frames entre gestos

### 🎨 Interfaz

- **Display principal**: Muestra el número/resultado actual
- **Expresión**: Muestra la operación completa
- **Indicador de gesto**: Panel grande mostrando el gesto detectado
- **Barra de confianza**: Indicador visual de la confianza del gesto
- **Guía lateral**: Lista completa de todos los gestos disponibles
- **Feedback**: Mensajes temporales de confirmación
- **Contador FPS**: En la esquina superior derecha

### 🐛 Notas Técnicas

- El sistema usa `eval()` para calcular expresiones (advertencia de seguridad en linting)
- Las variables no utilizadas en el código son intencionales para compatibilidad
- Los warnings de MediaPipe sobre "feedback tensors" son normales y no afectan el funcionamiento

## 📝 Requisitos

Ver `requirements.txt` para las dependencias necesarias.
