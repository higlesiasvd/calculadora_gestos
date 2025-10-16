#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculadora Gestual - Aplicación principal modularizada.

Aplicación de calculadora que se controla mediante gestos de manos,
utilizando MediaPipe para detección de gestos y OpenCV para renderizado.

Estructura modular:
    - config/: Configuración de accesibilidad
    - voice/: Sistema de feedback por voz
    - core/: Lógica principal (detección de gestos y calculadora)
    - ui/: Interfaz de usuario y renderizado
    - app/: Aplicación principal que integra todos los componentes

Autor: Sistema de IA con revisión humana
Fecha: Octubre 2025
Python: 3.11+
"""

# ============================================================================
# IMPORTS - Módulos del proyecto
# ============================================================================
from config.accessibility import AccessibilityConfig
from app.gesture_app import GestureCalculatorApp


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================
def main():
    """
    Punto de entrada de la aplicación.
    
    Manejo de errores:
        - KeyboardInterrupt (Ctrl+C): Cierre graceful por usuario
        - Exception general: Captura errores inesperados y muestra traceback
        
    Ejecución:
        python3 main.py
        
    Requisitos:
        - Python 3.11+
        - opencv-python==4.8.1.78
        - numpy==1.24.3
        - mediapipe
        - pyttsx3 (síntesis de voz)
        - Cámara conectada (índice 0)
        
    Características de accesibilidad:
        - Feedback por voz configurable (activar/desactivar con 'v')
        - Modo gestos extendidos para movilidad reducida (activar con 'a')
        - Guías visuales dinámicas cuando no detecta manos
    """
    try:
        # Crear configuración de accesibilidad
        config = AccessibilityConfig()
        
        # Configuraciones personalizables (descomenta para cambiar):
        # config.voice_enabled = True          # Activar voz por defecto
        # config.voice_volume = 0.8            # Volumen (0.0-1.0)
        # config.voice_rate = 150              # Velocidad de habla (palabras/min)
        # config.extended_gestures = False     # Modo gestos extendidos
        # config.show_hand_guides = True       # Mostrar guías de posicionamiento
        
        # Crear instancia de la aplicación (cámara índice 0)
        app = GestureCalculatorApp(camera_index=0, config=config)
        
        # Ejecutar bucle principal
        app.run()
    except KeyboardInterrupt:
        # Usuario presionó Ctrl+C
        print("\nInterrumpido por el usuario")
    except Exception as e:
        # Error inesperado - mostrar información completa
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
