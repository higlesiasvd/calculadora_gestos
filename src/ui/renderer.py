"""
Interfaz de usuario y renderizado.

Este módulo contiene la clase UIRenderer que dibuja todos los elementos visuales.
"""

import cv2
import numpy as np
import time


# ============================================================================
class UIRenderer:
    """
    Renderizador de interfaz gráfica para la calculadora gestual.
    
    Componentes visuales:
        1. Display principal: Muestra expresión, número actual y resultado
        2. Indicador de gesto: Muestra gesto detectado en tiempo real
        3. Guía lateral: Lista de gestos disponibles con descripciones
        4. Feedback: Mensajes temporales de confirmación/error
        5. Barra de cooldown: Progreso visual del tiempo de espera
        6. Guías de posicionamiento dinámicas (cuando no detecta bien)
    """
    
    def __init__(self, width, height, config=None):
        """
        Inicializa el renderizador con dimensiones de la ventana.
        
        Args:
            width (int): Ancho de la ventana en píxeles
            height (int): Alto de la ventana en píxeles
            config (AccessibilityConfig): Configuración de accesibilidad (opcional)
        """
        self.width = width
        self.height = height
        self.config = config if config else AccessibilityConfig()
        self.feedback_msg = ""               # Mensaje de feedback actual
        self.feedback_timer = 0              # Frames restantes para mostrar feedback
        self.feedback_color = (0, 255, 0)   # Color del feedback
        
        # Variables para guías visuales dinámicas
        self.no_detection_counter = 0        # Contador de frames sin detección
        self.show_positioning_guide = False  # Flag para mostrar guía de posicionamiento
    
    def show_feedback(self, msg, color=(0, 255, 0), duration=40):
        """
        Muestra mensaje de feedback temporal.
        
        Args:
            msg (str): Mensaje a mostrar
            color (tuple): Color BGR del mensaje
            duration (int): Duración en frames (~40 frames = 1.3 segundos @ 30fps)
            
        Usado para confirmar acciones (ej: "Número agregado", "Operación añadida")
        """
        self.feedback_msg = msg
        self.feedback_color = color
        self.feedback_timer = duration
    
    def draw_display(self, img, calc):
        """
        Dibuja el display principal de la calculadora.
        
        Args:
            img (np.array): Imagen sobre la cual dibujar
            calc (Calculator): Instancia de calculadora con estado actual
            
        Componentes:
            1. Fondo semi-transparente oscuro
            2. Título "CALCULADORA GESTUAL"
            3. Expresión matemática en construcción (parte superior)
            4. Display de número/resultado (grande, parte inferior)
            5. Cursor parpadeante cuando está esperando input
            
        Colores del display:
            - Blanco: Número normal
            - Verde: Resultado de cálculo
            - Rojo: Error
            
        Tamaños dinámicos:
            - Números cortos (<8 dígitos): Fuente 3.5
            - Números largos (≥8 dígitos): Fuente 2.5 (para que quepa)
        """
        x, y, w, h = 30, 30, self.width - 60, 220
        
        # Fondo semi-transparente usando overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (35, 35, 35), -1)
        cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)  # 92% opacidad
        cv2.rectangle(img, (x, y), (x + w, y + h), (100, 200, 255), 4)  # Borde azul
        
        # Título
        cv2.putText(img, "CALCULADORA GESTUAL", (x + 20, y + 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1.1, (200, 200, 200), 2)
        
        # Expresión matemática (parte superior del display)
        expr = calc.get_expression()
        if expr and expr != "0":
            cv2.putText(img, expr, (x + 20, y + 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
        
        # Display principal (número actual o resultado)
        display = calc.get_display()
        
        # Determinar color según estado
        color = (255, 255, 255)  # Blanco por defecto
        if display == "Error":
            color = (100, 100, 255)  # Rojo
        elif calc.result:
            color = (100, 255, 100)  # Verde (resultado)
        
        # Ajustar tamaño de fuente según longitud
        font_scale = 3.5 if len(display) < 8 else 2.5
        cv2.putText(img, display, (x + 20, y + 170),
                   cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 4)
        
        # Cursor parpadeante (solo si no hay resultado y está esperando input)
        if not calc.result and int(time.time() * 2) % 2 == 0:  # Parpadea a 1Hz
            text_w = cv2.getTextSize(display, cv2.FONT_HERSHEY_DUPLEX, font_scale, 4)[0][0]
            cx = x + 30 + text_w
            cv2.line(img, (cx, y + 130), (cx, y + 175), (0, 255, 0), 4)
    
    def draw_gesture(self, img, name, color, conf):
        if name in ["Sin mano", "Detectando...", "Estabilizando...", "..."]:
            return
        
        x, y, w, h = 50, 280, 600, 90
        
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.75 * conf, img, 1 - 0.75 * conf, 0, img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 4)
        
        cv2.putText(img, name, (x + 20, y + 60),
                   cv2.FONT_HERSHEY_DUPLEX, 1.9, (255, 255, 255), 3)
        
        bar_w = int((w - 40) * conf)
        cv2.rectangle(img, (x + 20, y + h - 15), (x + 20 + bar_w, y + h - 5),
                     (255, 255, 255), -1)
    
    def draw_guide(self, img):
        x, y = self.width - 500, 30
        w, h = 470, 900
        
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.90, img, 0.10, 0, img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 3)
        
        cv2.putText(img, "GESTOS INTUITIVOS", (x + 20, y + 45),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        
        guide = [
            ("NUMEROS", ""),
            ("  0: Puno cerrado", ""),
            ("  1-5: Dedos levantados", ""),
            ("  6-9: 5 dedos + extras", ""),
            ("", ""),
            ("OPERACIONES (2 manos)", ""),
            # Secciones con gestos actualizados según últimas mejoras
            ("  Suma: Cruz (indice+indice)", ""),
            ("  Multiplicar: X con indices", ""),
            ("  Dividir: Tijera (V + V)", ""),
            ("", ""),
            ("OPERACIONES (1 mano)", ""),
            ("  Resta: 4 dedos horizontal", ""),
            ("", ""),
            ("CONTROL", ""),
            ("  Pulgar arriba: = CALCULAR", ""),
            ("  Menique solo: Borrar todo", ""),
        ]
        
        cy = y + 90
        for label, _ in guide:
            # Línea vacía (espaciado)
            if not label:
                cy += 12
                continue
            
            # Encabezados de sección (mayúsculas, sin espacios iniciales)
            if label.isupper() and not label.startswith(" "):
                cv2.putText(img, label, (x + 20, cy),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 200, 255), 2)
                cy += 42
            # Elementos de lista (con espacios iniciales)
            else:
                cv2.putText(img, label, (x + 20, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
                cy += 35
    
    def draw_feedback(self, img):
        """
        Dibuja mensaje de feedback temporal en la parte inferior de la pantalla.
        
        Args:
            img (np.array): Imagen sobre la cual dibujar
            
        Efecto:
            - Aparece con fade-in/fade-out usando alpha blending
            - Duración controlada por feedback_timer
            - Color configurable (verde para éxito, rojo para error)
            
        Usado para confirmar acciones del usuario sin interrumpir flujo visual.
        """
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
            # Calcular alpha para fade-out suave
            alpha = min(self.feedback_timer / 20.0, 1.0)
            
            x, y = self.width // 2 - 250, self.height - 120
            
            # Fondo semi-transparente con fade
            overlay = img.copy()
            cv2.rectangle(overlay, (x - 20, y - 50), (x + 520, y + 10), (40, 40, 40), -1)
            cv2.addWeighted(overlay, alpha * 0.88, img, 1 - alpha * 0.88, 0, img)
            
            # Texto con alpha aplicado al color
            color = tuple(int(c * alpha) for c in self.feedback_color)
            cv2.putText(img, self.feedback_msg, (x, y),
                       cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3)
    
    def draw_cooldown(self, img, cd, max_cd):
        """
        Dibuja barra de cooldown (tiempo de espera entre gestos).
        
        Args:
            img (np.array): Imagen sobre la cual dibujar
            cd (int): Frames de cooldown restantes
            max_cd (int): Frames totales de cooldown
            
        Propósito:
            Prevenir procesamiento múltiple del mismo gesto.
            Usuario debe esperar hasta que la barra desaparezca para siguiente gesto.
        """
        if cd > 0:
            x, y = 50, 400
            # Barra proporcional al tiempo restante
            w = int(300 * (cd / max_cd))
            cv2.rectangle(img, (x, y), (x + w, y + 18), (255, 200, 0), -1)
            cv2.putText(img, "Procesando...", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
    
    def draw_positioning_guides(self, img, hands_data):
        """
        Dibuja guías visuales dinámicas para ayudar al posicionamiento de manos.
        
        Args:
            img (np.array): Imagen sobre la cual dibujar
            hands_data (list): Lista de manos detectadas
            
        Comportamiento:
            - Si no hay manos: Muestra guía central "Coloque su mano aquí"
            - Si hay mano pero muy lejos: Muestra "Acérquese más"
            - Si hay mano pero muy cerca: Muestra "Aléjese un poco"
            - Si posición correcta: No muestra guía
            
        Activación:
            - Se activa automáticamente tras 60 frames (~2s) sin detección válida
            - Se desactiva cuando se detecta gesto correctamente
        """
        if not self.config.show_hand_guides:
            return
        
        # Activar guía si no hay detección válida por mucho tiempo
        if len(hands_data) == 0:
            self.no_detection_counter += 1
            if self.no_detection_counter > 60:  # 2 segundos sin detección
                self.show_positioning_guide = True
        else:
            self.no_detection_counter = 0
            self.show_positioning_guide = False
        
        if not self.show_positioning_guide:
            return
        
        # ====================================================================
        # OVERLAY DE GUÍA: Rectángulo semi-transparente con instrucciones
        # ====================================================================
        center_x, center_y = self.width // 2, self.height // 2
        guide_w, guide_h = 400, 300
        
        # Fondo semi-transparente
        overlay = img.copy()
        cv2.rectangle(overlay, 
                     (center_x - guide_w // 2, center_y - guide_h // 2),
                     (center_x + guide_w // 2, center_y + guide_h // 2),
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, self.config.guide_opacity, img, 1 - self.config.guide_opacity, 0, img)
        
        # Borde de la guía
        cv2.rectangle(img,
                     (center_x - guide_w // 2, center_y - guide_h // 2),
                     (center_x + guide_w // 2, center_y + guide_h // 2),
                     (0, 200, 255), 3)
        
        # Icono de mano (símbolo simplificado)
        cv2.circle(img, (center_x, center_y - 50), 50, (100, 200, 255), 3)
        cv2.circle(img, (center_x, center_y - 50), 8, (100, 200, 255), -1)
        
        # Dedos simplificados
        for angle in [0, 30, 60, 90, 120]:
            end_x = int(center_x + 50 * np.cos(np.radians(angle - 90)))
            end_y = int(center_y - 50 + 50 * np.sin(np.radians(angle - 90)))
            cv2.line(img, (center_x, center_y - 50), (end_x, end_y), (100, 200, 255), 3)
        
        # Texto de instrucción
        cv2.putText(img, "COLOQUE SU MANO", (center_x - 150, center_y + 80),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img, "EN ESTA ZONA", (center_x - 120, center_y + 120),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        
        # Indicador parpadeante
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(img, (center_x, center_y + 150), 8, (0, 255, 0), -1)
    
    def update_detection_status(self, has_valid_gesture):
        """
        Actualiza el estado de detección para controlar guías visuales.
        
        Args:
            has_valid_gesture (bool): True si se detectó un gesto válido
        """
        if has_valid_gesture:
            self.no_detection_counter = 0
            self.show_positioning_guide = False

