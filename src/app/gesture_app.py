"""
Aplicación principal que integra todos los componentes.

Este módulo contiene la clase GestureCalculatorApp.
"""

import cv2
import time
from core.gesture_detector import GestureDetector
from core.calculator import Calculator
from ui.renderer import UIRenderer
from voice.feedback import VoiceFeedback
from config.accessibility import AccessibilityConfig


# ============================================================================
class GestureCalculatorApp:
    """
    Aplicación principal de calculadora gestual.
    
    Arquitectura:
        - GestureDetector: Detecta y reconoce gestos de manos
        - Calculator: Lógica aritmética y estado
        - UIRenderer: Renderizado de interfaz gráfica
        - GestureCalculatorApp: Coordinador y loop principal
        
    Sistema de hold-to-confirm:
        - Usuario debe mantener gesto durante 15 frames (~0.5s @ 30fps)
        - Barra de progreso visual muestra tiempo restante
        - Previene activaciones accidentales por movimientos rápidos
    """
    
    def __init__(self, camera_index=0, config=None):
        """
        Inicializa la aplicación y configura la cámara.
        
        Args:
            camera_index (int): Índice de la cámara (0 = cámara predeterminada)
            config (AccessibilityConfig): Configuración de accesibilidad (opcional)
            
        Configuración de cámara:
            - Resolución: 1920x1080 (Full HD)
            - FPS: 30 frames por segundo
            - Buffer: 1 frame (minimiza latencia)
            
        Raises:
            Exception: Si no se puede abrir la cámara
        """
        # Configuración de accesibilidad
        self.config = config if config else AccessibilityConfig()
        
        # Inicializar captura de video
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Error al abrir cámara")
        
        # Configurar parámetros de cámara para máxima calidad y mínima latencia
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Resolución horizontal
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Resolución vertical
        self.cap.set(cv2.CAP_PROP_FPS, 30)             # Frames por segundo
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # Buffer mínimo para baja latencia
        
        # Obtener dimensiones reales (pueden diferir de las solicitadas)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"OK Camara: {self.width}x{self.height}")
        
        # Inicializar componentes principales
        self.detector = GestureDetector()                              # Detector de gestos
        self.calc = Calculator()                                       # Lógica de calculadora
        self.ui = UIRenderer(self.width, self.height, self.config)    # Renderizador UI
        self.voice = VoiceFeedback(self.config)                       # Sistema de voz
        
        # Variables de control de gestos
        self.last_gesture = "none"      # Último gesto procesado (previene duplicados)
        self.cooldown = 0               # Frames restantes de cooldown
        self.cooldown_time = 25         # Cooldown estándar (~0.8s @ 30fps)
        
        # Variables de FPS (frames por segundo)
        self.fps_time = time.time()
        self.fps = 0
        
        # Mostrar configuración de accesibilidad
        if self.config.extended_gestures:
            print("✓ Modo Gestos Extendidos ACTIVADO (para movilidad reducida)")
        if self.config.voice_enabled:
            print("✓ Feedback por voz ACTIVADO")
    
    def process(self, gid, gname):
        """
        Procesa un gesto detectado y actualiza el estado de la calculadora.
        
        Args:
            gid (str): ID del gesto (ej: "num_5", "add", "equal")
            gname (str): Nombre del gesto (para display, no usado actualmente)
            
        Cooldown:
            - Previene procesamiento múltiple del mismo gesto
            - Usuario debe esperar ~0.8 segundos entre gestos
            - Se resetea el buffer del detector tras cada acción exitosa
            
        NUEVO: Feedback por voz
            - Cada acción reproduce un mensaje de voz confirmando la acción
            
        Feedback:
            - Cada acción muestra mensaje de confirmación
            - Color verde: Números y suma
            - Color naranja: Resta y operaciones
            - Color cian: Resultado de cálculo
            - Color rojo: Error o borrado
        """
        # Ignorar si aún hay cooldown o es el mismo gesto que antes
        if self.cooldown > 0 or gid == self.last_gesture:
            return
        
        self.last_gesture = gid
        
        # ====================================================================
        # NÚMEROS (0-9): Añadir dígito a número actual
        # ====================================================================
        if gid.startswith("num_"):
            digit = int(gid.split("_")[1])  # Extraer número del ID
            if self.calc.add_digit(digit):
                self.ui.show_feedback(f"OK {digit}", (100, 255, 100))
                self.voice.speak_number(digit)  # Feedback por voz
                self.cooldown = self.cooldown_time
                self.detector.reset_buffer()
        
        # ====================================================================
        # SUMA (+): Añadir operador de suma
        # ====================================================================
        elif gid == "add":
            if self.calc.add_operation("+"):
                self.ui.show_feedback("+ SUMA", (0, 255, 0))
                self.voice.speak_operation("+")  # Feedback por voz
                self.cooldown = self.cooldown_time
                self.detector.reset_buffer()
        
        # ====================================================================
        # RESTA (-): Añadir operador de resta
        # ====================================================================
        elif gid == "subtract":
            if self.calc.add_operation("-"):
                self.ui.show_feedback("- RESTA", (255, 150, 0))
                self.voice.speak_operation("-")  # Feedback por voz
                self.cooldown = self.cooldown_time
                self.detector.reset_buffer()
        
        # ====================================================================
        # MULTIPLICACIÓN (×): Añadir operador de multiplicación
        # ====================================================================
        elif gid == "multiply":
            if self.calc.add_operation("*"):
                self.ui.show_feedback("x MULTIPLICAR", (255, 100, 255))
                self.voice.speak_operation("*")  # Feedback por voz
                self.cooldown = self.cooldown_time
                self.detector.reset_buffer()
        
        # ====================================================================
        # DIVISIÓN (÷): Añadir operador de división
        # ====================================================================
        elif gid == "divide":
            if self.calc.add_operation("/"):
                self.ui.show_feedback("/ DIVIDIR", (150, 100, 255))
                self.voice.speak_operation("/")  # Feedback por voz
                self.cooldown = self.cooldown_time
                self.detector.reset_buffer()
        
        # ====================================================================
        # IGUAL (=): Calcular resultado de la expresión
        # ====================================================================
        elif gid == "equal":
            success, result = self.calc.calculate()
            if success:
                self.ui.show_feedback(f"= {result}", (0, 255, 255), 60)
                self.voice.speak_result(result)  # Feedback por voz: "igual a X"
            else:
                self.ui.show_feedback("Error", (255, 50, 50))
                self.voice.speak("error de cálculo")  # Feedback por voz
            self.cooldown = self.cooldown_time
            self.detector.reset_buffer()
        
        # ====================================================================
        # BORRAR TODO (C): Resetear calculadora completamente
        # ====================================================================
        elif gid == "clear_all":
            self.calc.clear_all()
            self.ui.show_feedback("TODO BORRADO", (255, 50, 50))
            self.voice.speak("todo borrado")  # Feedback por voz
            self.cooldown = self.cooldown_time
            self.detector.reset_buffer()
        
        # ====================================================================
        # BACKSPACE (←): Borrar último carácter (actualmente sin gesto)
        # ====================================================================
        elif gid == "backspace":
            self.calc.backspace()
            self.ui.show_feedback("← BORRADO", (255, 200, 0))
            self.cooldown = self.cooldown_time // 2  # Cooldown más corto
            self.detector.reset_buffer()
    
    def run(self):
        """
        Bucle principal de la aplicación.
        
        Ciclo de ejecución:
            1. Capturar frame de cámara
            2. Espejear frame (flip horizontal para UI natural)
            3. Detectar landmarks de manos con MediaPipe
            4. Detectar gesto estable (con buffer)
            5. Procesar gesto si cumple requisitos
            6. Renderizar UI completa
            7. Mostrar frame y procesar input de teclado
            8. Repetir hasta ESC o 'q'
            
        Controles de teclado:
            - ESC o 'q': Salir de la aplicación
            
        Sistema de hold-to-confirm:
            - Usuario debe mantener gesto durante 15 frames (~0.5s)
            - Barra de progreso muestra tiempo restante
            - Al completar, gesto se procesa y entra en cooldown
        """
        # Imprimir instrucciones de uso en terminal
        print("\n" + "="*70)
        print("CALCULADORA GESTUAL - GESTOS MEJORADOS")
        print("="*70)
        print("\nNumeros: 0-5 dedos levantados")
        print("Suma: Cruz con indices (2 manos)")
        print("Resta: 4 dedos horizontales (1 mano)")
        print("Multiplicar: X con indices (2 manos)")
        print("Dividir: Tijera V+V (2 manos)")
        print("Calcular: Pulgar arriba")
        
        # Instrucciones de accesibilidad
        if self.config.voice_enabled:
            print("\n🔊 FEEDBACK POR VOZ: Activado")
        if self.config.extended_gestures:
            print("♿ MODO ACCESIBILIDAD: Gestos extendidos activados")
        if self.config.show_hand_guides:
            print("👁 AYUDAS VISUALES: Guías de posicionamiento activadas")
        
        print("\nPresiona ESC o 'q' para salir")
        print("Presiona 'v' para activar/desactivar voz")
        print("Presiona 'a' para activar/desactivar modo accesibilidad\n")
        print("="*70 + "\n")
        
        # ====================================================================
        # BUCLE PRINCIPAL - Se ejecuta aproximadamente a 30 FPS
        # ====================================================================
        while True:
            # Capturar frame de la cámara
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Espejear horizontalmente para UI más intuitiva
            # (usuario mueve mano derecha, aparece a la derecha)
            frame = cv2.flip(frame, 1)
            
            # ================================================================
            # DETECCIÓN DE GESTOS
            # ================================================================
            hands_data, results = self.detector.get_landmarks(frame)
            frame = self.detector.draw_hands(frame, results)
            
            gid, gname, conf, color = self.detector.detect_gesture_stable(hands_data)
            
            # Actualizar estado de detección para guías visuales
            has_valid_gesture = conf > 0.7 and gid not in ["none", "unknown"]
            self.ui.update_detection_status(has_valid_gesture)
            
            # ================================================================
            # PROCESAMIENTO DE GESTOS
            # Solo procesa gestos con confianza > 70%
            # ================================================================
            if conf > 0.7:
                self.process(gid, gname)
            
            # Decrementar cooldown cada frame (cuenta regresiva)
            if self.cooldown > 0:
                self.cooldown -= 1
            
            # ================================================================
            # RENDERIZADO DE UI
            # Dibuja todos los componentes visuales sobre el frame
            # ================================================================
            self.ui.draw_display(frame, self.calc)        # Display principal
            self.ui.draw_gesture(frame, gname, color, conf)  # Indicador de gesto
            self.ui.draw_guide(frame)                      # Guía lateral
            self.ui.draw_feedback(frame)                   # Feedback temporal
            self.ui.draw_cooldown(frame, self.cooldown, self.cooldown_time)  # Barra cooldown
            self.ui.draw_positioning_guides(frame, hands_data)  # NUEVO: Guías dinámicas
            
            # ================================================================
            # CÁLCULO Y DISPLAY DE FPS
            # ================================================================
            current_time = time.time()
            self.fps = 1 / (current_time - self.fps_time + 1e-6)  # +epsilon evita div/0
            self.fps_time = current_time
            
            cv2.putText(frame, f"FPS: {int(self.fps)}", (self.width - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # ================================================================
            # INDICADORES DE ACCESIBILIDAD
            # ================================================================
            y_offset = 70
            if self.config.voice_enabled:
                cv2.putText(frame, "VOZ: ON", (self.width - 150, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 30
            if self.config.extended_gestures:
                cv2.putText(frame, "MODO ACCESIBLE", (self.width - 220, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            
            # ================================================================
            # INSTRUCCIONES DE SALIDA
            # ================================================================
            cv2.putText(frame, "ESC/q: salir | v: voz | a: accesibilidad", 
                       (50, self.height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # ================================================================
            # MOSTRAR FRAME Y PROCESAR INPUT
            # ================================================================
            cv2.imshow('Calculadora Gestual', frame)
            
            # Esperar 1ms y procesar teclas presionadas
            key = cv2.waitKey(1) & 0xFF
            
            # ESC o 'q' para salir
            if key == 27 or key == ord('q'):
                break
            
            # 'v' para activar/desactivar voz
            elif key == ord('v'):
                self.config.voice_enabled = not self.config.voice_enabled
                status = "ACTIVADA" if self.config.voice_enabled else "DESACTIVADA"
                print(f"🔊 Voz: {status}")
                self.ui.show_feedback(f"VOZ {status}", (0, 255, 255), 60)
                if self.config.voice_enabled:
                    self.voice.speak("voz activada")
                else:
                    # Decir "voz desactivada" antes de apagarse
                    self.voice.speak("voz desactivada")
            
            # 'a' para activar/desactivar modo accesibilidad
            elif key == ord('a'):
                self.config.extended_gestures = not self.config.extended_gestures
                status = "ACTIVADO" if self.config.extended_gestures else "DESACTIVADO"
                print(f"♿ Modo Accesibilidad: {status}")
                self.ui.show_feedback(f"ACCESIBILIDAD {status}", (255, 200, 0), 60)
                if self.config.voice_enabled:
                    mensaje = "modo accesible activado" if self.config.extended_gestures else "modo accesible desactivado"
                    self.voice.speak(mensaje)
        
        # ====================================================================
        # LIMPIEZA Y CIERRE
        # ====================================================================
        self.cap.release()          # Liberar cámara
        cv2.destroyAllWindows()     # Cerrar ventanas de OpenCV
        print("\nOK Aplicacion cerrada correctamente")

