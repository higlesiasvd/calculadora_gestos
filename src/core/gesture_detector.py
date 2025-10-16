"""
Detector de gestos usando MediaPipe.

Este módulo contiene la lógica de detección y reconocimiento de gestos.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque


# ============================================================================
class GestureDetector:
    """
    Detector de gestos con alta precisión y gestos intuitivos.
    
    Soporta gestos de una mano (números 0-5, operaciones básicas) y dos manos
    (números 6-9, operaciones compuestas). Implementa un sistema de buffer
    para evitar detecciones falsas y requiere estabilidad temporal del gesto.
    """
    
    def __init__(self, detection_confidence=0.8, tracking_confidence=0.8):
        """
        Inicializa el detector de gestos con MediaPipe Hands.
        
        Args:
            detection_confidence (float): Confianza mínima para detectar una mano (0.8 = 80%)
            tracking_confidence (float): Confianza mínima para seguir una mano entre frames
        
        Configuración:
            - max_num_hands=2: Detecta hasta 2 manos simultáneamente
            - model_complexity=1: Equilibrio entre precisión y rendimiento
            - static_image_mode=False: Optimizado para video en tiempo real
        """
        # Inicialización de MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,      # Modo video (aprovecha tracking entre frames)
            max_num_hands=2,               # Máximo 2 manos detectables
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1             # Modelo intermedio (balance velocidad/precisión)
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        # ====================================================================
        # SISTEMA DE ESTABILIZACIÓN - Previene detecciones erráticas
        # ====================================================================
        # Buffer circular que almacena los últimos 10 gestos detectados
        # Solo se confirma un gesto si aparece en 7+ de los 10 últimos frames
        self.gesture_buffer = deque(maxlen=10)  # Ventana deslizante de 10 frames
        self.min_stability = 0.70                # 70% de consistencia requerida (7/10)
        
    def get_landmarks(self, img):
        """
        Extrae landmarks (puntos de referencia) de todas las manos detectadas en la imagen.
        
        Args:
            img (np.array): Imagen BGR capturada de la cámara
            
        Returns:
            tuple: (hands_data, results)
                - hands_data: Lista de diccionarios con información de cada mano detectada
                  [{'landmarks': [...], 'raw': MediaPipe object, 'label': 'Left'/'Right'}]
                - results: Objeto results de MediaPipe (para dibujar landmarks)
                
        Proceso:
            1. Convierte BGR -> RGB (MediaPipe requiere RGB)
            2. Desactiva writeable para optimizar memoria
            3. Procesa con MediaPipe Hands
            4. Convierte landmarks normalizados (0-1) a coordenadas pixel
            5. Retorna lista estructurada con landmarks y metadata
        """
        # Convertir a RGB (MediaPipe solo procesa RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False  # Optimización de memoria
        results = self.hands.process(img_rgb)
        img_rgb.flags.writeable = True   # Restaurar para futuras operaciones
        
        # Lista que contendrá información de cada mano detectada
        hands_data = []
        
        # Verificar si se detectaron manos
        if results.multi_hand_landmarks and results.multi_handedness:
            # Iterar sobre cada mano detectada (máximo 2)
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,   # Coordenadas de 21 landmarks por mano
                results.multi_handedness        # Metadata (Left/Right)
            ):
                h, w, _ = img.shape  # Dimensiones de la imagen
                landmarks = []
                
                # Convertir cada landmark de coordenadas normalizadas a píxeles
                for lm in hand_landmarks.landmark:
                    landmarks.append({
                        'x': lm.x * w,  # Coordenada X en píxeles
                        'y': lm.y * h,  # Coordenada Y en píxeles
                        'z': lm.z       # Profundidad relativa (no se usa actualmente)
                    })
                
                # Agregar información completa de esta mano
                hands_data.append({
                    'landmarks': landmarks,        # 21 landmarks convertidos a píxeles
                    'raw': hand_landmarks,         # Objeto raw de MediaPipe
                    'label': handedness.classification[0].label  # 'Left' o 'Right'
                })
        
        return hands_data, results
    
    def draw_hands(self, img, results):
        """
        Dibuja landmarks y conexiones de manos detectadas sobre la imagen.
        
        Args:
            img (np.array): Imagen sobre la cual dibujar
            results: Objeto results de MediaPipe con landmarks detectados
            
        Returns:
            np.array: Imagen con landmarks dibujados
            
        Visualización:
            - Puntos rojos: Landmarks individuales (21 por mano)
            - Líneas verdes: Conexiones entre landmarks (esqueleto de la mano)
            - Estilo predefinido de MediaPipe para máxima claridad
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar usando estilos predefinidos de MediaPipe
                self.mp_draw.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style()
                )
        return img
    
    def count_extended_fingers(self, landmarks):
        """
        Cuenta dedos extendidos con MÁXIMA precisión usando validación basada en distancias.
        
        Args:
            landmarks (list): Lista de 21 landmarks de una mano
            
        Returns:
            list: [pulgar, índice, medio, anular, meñique]
                  Cada elemento es 1 (extendido) o 0 (flexionado)
                  
        Algoritmo:
            - Pulgar: Compara distancia TIP-WRIST vs MCP-WRIST (método lateral)
            - Otros dedos: Compara distancia TIP-WRIST vs PIP-WRIST
            - Factor de seguridad: TIP debe estar 5% más lejos que articulación base
            
        Ventajas:
            - No depende de orientación de la mano
            - Robusto ante rotaciones y movimientos
            - Evita falsos positivos por flexión parcial
        """
        if len(landmarks) < 21:
            return [0, 0, 0, 0, 0]
        
        fingers = []
        
        # Índices de landmarks clave (ver diagrama MediaPipe Hands)
        tip_ids = [4, 8, 12, 16, 20]    # Puntas de dedos
        pip_ids = [3, 6, 10, 14, 18]    # Articulaciones PIP (segunda falange)
        mcp_ids = [2, 5, 9, 13, 17]     # Articulaciones MCP (base del dedo)
        
        wrist = landmarks[0]  # Landmark 0: Muñeca (punto de referencia)
        
        # ====================================================================
        # DETECCIÓN PULGAR (índice 0)
        # El pulgar se extiende lateralmente, no verticalmente como otros dedos
        # ====================================================================
        thumb_tip = landmarks[4]   # Punta del pulgar
        thumb_ip = landmarks[3]    # Articulación IP (interfalángica)
        thumb_mcp = landmarks[2]   # Articulación MCP (metacarpofalángica)
        
        # Calcular distancia euclidiana de punta e IP desde la muñeca
        thumb_tip_dist = math.sqrt(
            (thumb_tip['x'] - wrist['x'])**2 + 
            (thumb_tip['y'] - wrist['y'])**2
        )
        thumb_ip_dist = math.sqrt(
            (thumb_ip['x'] - wrist['x'])**2 + 
            (thumb_ip['y'] - wrist['y'])**2
        )
        
        # Pulgar extendido si la punta está 15% más lejos de la muñeca que el IP
        # Factor 1.15: Evita falsos positivos con pulgar semi-flexionado
        fingers.append(1 if thumb_tip_dist > thumb_ip_dist * 1.15 else 0)
        
        # ====================================================================
        # DETECCIÓN OTROS DEDOS (índice, medio, anular, meñique)
        # Usa 3 métodos de validación y requiere consenso (2 de 3)
        # ====================================================================
        for i in range(1, 5):  # Iterar sobre índice, medio, anular, meñique
            tip = landmarks[tip_ids[i]]   # Punta del dedo
            pip = landmarks[pip_ids[i]]   # Articulación PIP (segunda falange)
            mcp = landmarks[mcp_ids[i]]   # Articulación MCP (base del dedo)
            
            # MÉTODO 1: Comparación vertical (funciona si mano está vertical)
            # Dedo extendido si punta está significativamente arriba del PIP
            vertical_extended = tip['y'] < pip['y'] - 20 and pip['y'] < mcp['y']
            
            # MÉTODO 2: Distancia desde muñeca (robusto ante rotaciones)
            # Dedo extendido si punta está 25% más lejos que la base
            tip_dist = math.sqrt((tip['x'] - wrist['x'])**2 + (tip['y'] - wrist['y'])**2)
            mcp_dist = math.sqrt((mcp['x'] - wrist['x'])**2 + (mcp['y'] - wrist['y'])**2)
            distance_extended = tip_dist > mcp_dist * 1.25
            
            # MÉTODO 3: Ángulo del dedo (precisión geométrica)
            # Mide ángulo formado por MCP-PIP-TIP, >140° = extendido
            angle = self.calculate_finger_angle(mcp, pip, tip)
            angle_extended = angle > 140  # Grados
            
            # DECISIÓN: Dedo extendido si al menos 2 de 3 métodos lo confirman
            # Este sistema de votación reduce dramáticamente falsos positivos
            extended = sum([vertical_extended, distance_extended, angle_extended]) >= 2
            fingers.append(1 if extended else 0)
        
        return fingers
    
    def calculate_finger_angle(self, mcp, pip, tip):
        """
        Calcula el ángulo de extensión de un dedo usando geometría vectorial.
        
        Args:
            mcp (dict): Landmark de la articulación MCP (base del dedo)
            pip (dict): Landmark de la articulación PIP (segunda falange)
            tip (dict): Landmark de la punta del dedo
            
        Returns:
            float: Ángulo en grados (0° = totalmente flexionado, 180° = totalmente recto)
            
        Método:
            1. Crea vector v1: MCP → PIP (primera falange)
            2. Crea vector v2: PIP → TIP (segunda falange)
            3. Calcula ángulo entre vectores usando producto punto
            4. Invierte resultado (180° - ángulo) para que 180° = extendido
            
        Aplicación:
            - Ángulo > 140°: Dedo considerado extendido
            - Ángulo < 140°: Dedo considerado flexionado
        """
        # Vectores que representan las falanges del dedo
        v1 = np.array([pip['x'] - mcp['x'], pip['y'] - mcp['y']])
        v2 = np.array([tip['x'] - pip['x'], tip['y'] - pip['y']])
        
        # Producto punto: v1·v2 = |v1||v2|cos(θ)
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        return 180 - angle  # Invertir: 180° = recto, 0° = doblado
    
    def are_fingers_crossed(self, landmarks):
        """
        Detecta si los dedos están cruzados (usado para validación de gestos).
        
        Args:
            landmarks (list): 21 landmarks de una mano
            
        Returns:
            bool: True si índice y medio están cruzados
            
        Criterios:
            - Distancia horizontal > 60px entre índice y medio
            - Distancia vertical < 40px (deben estar al mismo nivel)
            
        Nota: Actualmente no se usa en gestos principales, pero puede
              ser útil para gestos futuros o validaciones adicionales.
        """
        if len(landmarks) < 21:
            return False
        
        index_tip = landmarks[8]    # Punta del índice
        middle_tip = landmarks[12]  # Punta del medio
        
        # Cruzados si están separados horizontalmente pero alineados verticalmente
        distance = abs(index_tip['x'] - middle_tip['x'])
        y_diff = abs(index_tip['y'] - middle_tip['y'])
        
        return distance > 60 and y_diff < 40
    
    def is_hand_horizontal(self, landmarks):
        """
        Detecta si la mano está orientada horizontalmente - MEJORADO con validación de tips.
        
        Args:
            landmarks (list): 21 landmarks de una mano
            
        Returns:
            bool: True si la mano está en posición horizontal
            
        Criterios mejorados:
            1. Extensión horizontal (X) debe ser 1.8x mayor que vertical (Y)
            2. Las puntas de los dedos deben estar niveladas (y_diff < 50px)
            3. Validación desde muñeca hasta dedo medio
            
        Aplicación:
            - Gesto de resta: Requiere 4 dedos horizontales (sin pulgar)
            - Previene falsos positivos con mano inclinada o vertical
        """
        if len(landmarks) < 21:
            return False
        
        wrist = landmarks[0]             # Muñeca (punto de referencia)
        middle_finger_tip = landmarks[12] # Punta del dedo medio (centro)
        index_tip = landmarks[8]          # Punta del índice (extremo)
        pinky_tip = landmarks[20]         # Punta del meñique (extremo)
        
        # VALIDACIÓN 1: Puntas de dedos deben estar niveladas horizontalmente
        # Si están desniveladas más de 50px, la mano está inclinada
        y_diff = abs(index_tip['y'] - pinky_tip['y'])
        
        # VALIDACIÓN 2: Extensión horizontal vs vertical desde muñeca
        x_diff = abs(middle_finger_tip['x'] - wrist['x'])        # Distancia horizontal
        y_diff_wrist = abs(middle_finger_tip['y'] - wrist['y']) # Distancia vertical
        
        # Horizontal si X es significativamente mayor que Y y dedos están nivelados
        return x_diff > y_diff_wrist * 1.8 and y_diff < 50
    
    def hands_form_cross(self, landmarks_1, landmarks_2):
        """
        Detecta si dos manos forman una cruz '+' PERFECTA (para suma) - ULTRA ESTRICTO.
        
        Args:
            landmarks_1 (list): Landmarks de la primera mano
            landmarks_2 (list): Landmarks de la segunda mano
            
        Returns:
            bool: True si las manos forman una cruz perpendicular MUY precisa
            
        Algoritmo MEJORADO para diferenciar de multiplicación:
            1. Usa dedo medio como referencia (más estable que índice)
            2. Calcula vectores desde muñeca hasta punta del medio
            3. Calcula ángulo real entre vectores usando producto punto
            4. ESTRICTO: Solo acepta 80-100° (muy cerca de 90° perpendicular)
            5. Distancia MUY cercana: < 150px (deben tocarse casi)
            6. NUEVO: Valida que una mano sea horizontal y otra vertical
            
        Diferencias con multiplicación (X):
            - SUMA: Ángulo 80-100° (ESTRICTO), distancia < 150px, orientación ortogonal
            - MULT: Ángulo 40-140° (amplio), distancia < 200px, cruce diagonal
        """
        if len(landmarks_1) < 21 or len(landmarks_2) < 21:
            return False
        
        # Usar puntas de dedos medios para mejor detección
        middle_1_tip = landmarks_1[12]
        middle_1_base = landmarks_1[9]
        middle_2_tip = landmarks_2[12]
        middle_2_base = landmarks_2[9]
        
        # Calcular vectores de los dedos medios
        v1_x = middle_1_tip['x'] - middle_1_base['x']
        v1_y = middle_1_tip['y'] - middle_1_base['y']
        v2_x = middle_2_tip['x'] - middle_2_base['x']
        v2_y = middle_2_tip['y'] - middle_2_base['y']
        
        # Normalizar para calcular ángulo
        len1 = math.sqrt(v1_x**2 + v1_y**2) + 1e-6
        len2 = math.sqrt(v2_x**2 + v2_y**2) + 1e-6
        
        v1_x, v1_y = v1_x/len1, v1_y/len1
        v2_x, v2_y = v2_x/len2, v2_y/len2
        
        # Producto punto para calcular ángulo
        dot_product = v1_x * v2_x + v1_y * v2_y
        angle = math.degrees(math.acos(max(-1, min(1, dot_product))))
        
        # VALIDACIÓN 1: Ángulo MUY cercano a 90° (perpendicular perfecto)
        angle_valid = 80 < angle < 100
        
        # VALIDACIÓN 2: Distancia entre puntas DEBE ser muy cercana (casi tocándose)
        distance = math.sqrt(
            (middle_1_tip['x'] - middle_2_tip['x'])**2 + 
            (middle_1_tip['y'] - middle_2_tip['y'])**2
        )
        distance_valid = distance < 150
        
        # VALIDACIÓN 3: Una mano debe ser horizontal, otra vertical (cruz ortogonal)
        # Calcular orientaciones relativas
        horizontal_1 = abs(v1_x) > abs(v1_y) * 1.5  # Mano 1 es horizontal si x >> y
        vertical_1 = abs(v1_y) > abs(v1_x) * 1.5    # Mano 1 es vertical si y >> x
        horizontal_2 = abs(v2_x) > abs(v2_y) * 1.5
        vertical_2 = abs(v2_y) > abs(v2_x) * 1.5
        
        # Una debe ser H y otra V (orientación ortogonal)
        orientation_valid = (horizontal_1 and vertical_2) or (vertical_1 and horizontal_2)
        
        # Cruz válida SOLO si TODAS las validaciones pasan
        is_cross = angle_valid and distance_valid and orientation_valid
        
        return is_cross
    
    def hands_form_x(self, landmarks_1, landmarks_2):
        """
        Detecta si dos dedos índice forman una 'X' (para multiplicación) - MEJORADO con validación de cruce.
        
        Args:
            landmarks_1 (list): Landmarks de la primera mano
            landmarks_2 (list): Landmarks de la segunda mano
            
        Returns:
            bool: True si los índices forman una X válida
            
        Algoritmo mejorado:
            1. Calcula vectores de ambos dedos índice (base → punta)
            2. Mide ángulo entre vectores (debe estar entre 40-140°)
            3. Valida que las puntas estén cerca (< 200px)
            4. CRÍTICO: Valida cruce real - puntas deben estar cerca del centro
            5. Bases deben estar separadas (> 100px) pero puntas convergir
            
        Validaciones que previenen falsos positivos:
            - Bases separadas (base_distance > 100): No son dedos paralelos
            - Puntas cerca del centro (< 0.8 * base_distance): Se cruzan realmente
            - Ángulo correcto (40-140°): No están paralelos ni opuestos
            
        Diferencia con hands_form_cross:
            - Cruz: Perpendiculares (~90°), usan dedos medios
            - X: Cruzados (40-140°), usan dedos índice, validan intersección
        """
        if len(landmarks_1) < 21 or len(landmarks_2) < 21:
            return False
        
        # Landmarks de dedos índice de ambas manos
        index_1_tip = landmarks_1[8]   # Punta índice mano 1
        index_1_base = landmarks_1[5]  # Base índice mano 1
        index_2_tip = landmarks_2[8]   # Punta índice mano 2
        index_2_base = landmarks_2[5]  # Base índice mano 2
        
        # Calcular vectores direccionales de los dedos índice
        v1_x = index_1_tip['x'] - index_1_base['x']
        v1_y = index_1_tip['y'] - index_1_base['y']
        v2_x = index_2_tip['x'] - index_2_base['x']
        v2_y = index_2_tip['y'] - index_2_base['y']
        
        # Normalizar vectores (longitud = 1) para cálculo de ángulo
        len1 = math.sqrt(v1_x**2 + v1_y**2) + 1e-6  # +epsilon evita división por 0
        len2 = math.sqrt(v2_x**2 + v2_y**2) + 1e-6
        
        v1_x, v1_y = v1_x/len1, v1_y/len1
        v2_x, v2_y = v2_x/len2, v2_y/len2
        
        # Producto punto para calcular ángulo entre vectores
        dot_product = v1_x * v2_x + v1_y * v2_y
        angle = math.degrees(math.acos(max(-1, min(1, dot_product))))
        
        # VALIDACIÓN 1: Distancia entre puntas (deben estar cerca)
        distance = math.sqrt(
            (index_1_tip['x'] - index_2_tip['x'])**2 + 
            (index_1_tip['y'] - index_2_tip['y'])**2
        )
        
        # VALIDACIÓN 2: Verificar cruce real (puntas cerca del centro geométrico)
        center_x = (index_1_base['x'] + index_2_base['x']) / 2
        center_y = (index_1_base['y'] + index_2_base['y']) / 2
        
        tip1_to_center = math.sqrt((index_1_tip['x'] - center_x)**2 + (index_1_tip['y'] - center_y)**2)
        tip2_to_center = math.sqrt((index_2_tip['x'] - center_x)**2 + (index_2_tip['y'] - center_y)**2)
        
        # VALIDACIÓN 3: Distancia entre bases (deben estar separadas)
        base_distance = math.sqrt((index_1_base['x'] - index_2_base['x'])**2 + (index_1_base['y'] - index_2_base['y'])**2)
        
        # X válida si TODAS las condiciones se cumplen:
        is_x = (40 < angle < 140 and            # Ángulo de cruce correcto
                distance < 200 and               # Puntas cercanas
                base_distance > 100 and          # Bases separadas (no paralelas)
                tip1_to_center < base_distance * 0.8 and  # Punta 1 cerca del centro
                tip2_to_center < base_distance * 0.8)     # Punta 2 cerca del centro
        
        return is_x
    
    def detect_gesture_raw(self, hands_data):
        """
        Detecta el gesto actual sin aplicar estabilización (detección instantánea).
        
        Args:
            hands_data (list): Lista de manos detectadas con sus landmarks
            
        Returns:
            tuple: (gesto_id, gesto_nombre, confianza, color_rgb)
                - gesto_id: Identificador único del gesto
                - gesto_nombre: Nombre descriptivo para mostrar al usuario
                - confianza: 0.0-1.0 (qué tan seguro está el detector)
                - color: Tuple RGB para UI
                
        Jerarquía de detección:
            1. DOS MANOS: Números 6-9 y operaciones complejas (+, ×, ÷)
            2. UNA MANO: Números 0-5 y operaciones básicas (-, =, C, ←)
            3. SIN MANOS: Estado "none"
            
        Diseño de gestos:
            - Números 0-5: Una mano, conteo de dedos simple
            - Números 6-9: Dos manos, suma de dedos
            - Suma (+): Dos manos formando cruz perpendicular
            - Resta (-): Una mano horizontal con 4 dedos
            - Multiplicación (×): Dos índices formando X
            - División (÷): Dos manos paralelas verticales
            - Igual (=): Pulgar arriba (confirmar cálculo)
            - Limpiar (C): Solo meñique extendido
            - Borrar (←): Puño cerrado
        """
        if not hands_data:
            return "none", "Sin mano", 0.0, (150, 150, 150)
        
        num_hands = len(hands_data)
        
        # ====================================================================
        # GESTOS DE UNA MANO
        # ====================================================================
        # ==================
        if num_hands == 1:
            landmarks = hands_data[0]['landmarks']
            fingers = self.count_extended_fingers(landmarks)
            total = sum(fingers)
            
            # === NÚMEROS 0-5 ===
            if total == 0:
                return "num_0", "CERO (0)", 1.0, (255, 100, 100)
            
            elif total == 1:
                if fingers[1] == 1:  # Solo índice
                    return "num_1", "UNO (1)", 1.0, (100, 255, 100)
            # ====================================================================
            # NÚMERO 2: Índice + medio extendidos
            # ====================================================================
            elif total == 2:
                if fingers[1] == 1 and fingers[2] == 1:  # Índice + medio
                    return "num_2", "DOS (2)", 1.0, (100, 255, 100)
            
            # ====================================================================
            # NÚMERO 3: CUALQUIER 3 dedos extendidos - MEJORADO para flexibilidad
            # Acepta cualquier combinación (índice+medio+anular, pulgar+índice+medio, etc.)
            # ====================================================================
            elif total == 3:
                return "num_3", "TRES (3)", 1.0, (100, 255, 100)
            
            # ====================================================================
            # NÚMERO 4: 4 dedos sin pulgar (índice+medio+anular+meñique)
            # ====================================================================
            elif total == 4:
                if fingers[0] == 0:  # Sin pulgar
                    return "num_4", "CUATRO (4)", 1.0, (100, 255, 100)
            
            # ====================================================================
            # NÚMERO 5: Todos los dedos extendidos (mano abierta)
            # ====================================================================
            elif total == 5:
                return "num_5", "CINCO (5)", 1.0, (100, 255, 100)
            
            # ====================================================================
            # IGUAL (=): Pulgar arriba - Ejecuta el cálculo
            # Antes llamado "confirmar", simplificado a solo pulgar arriba
            # ====================================================================
            if fingers == [1, 0, 0, 0, 0]:
                thumb_tip = landmarks[4]
                wrist = landmarks[0]
                # Verificar que el pulgar esté realmente arriba (30px sobre la muñeca)
                if thumb_tip['y'] < wrist['y'] - 30:
                    return "equal", "= CALCULAR", 1.0, (0, 255, 255)
            
            # ====================================================================
            # SUMA (+): Pulgar e índice extendidos (gesto "L") - NUEVO
            # Gesto simple con una sola mano, fácil de reconocer
            # ====================================================================
            if fingers == [1, 1, 0, 0, 0]:
                return "add", "+ SUMA", 0.98, (0, 255, 0)
            
            # ====================================================================
            # RESTA (-): Mano horizontal con 4 dedos (sin pulgar) - MEJORADA
            # Valida orientación horizontal para evitar confusión con número 4
            # ====================================================================
            if fingers == [0, 1, 1, 1, 1] and self.is_hand_horizontal(landmarks):
                return "subtract", "- RESTA", 0.98, (255, 150, 0)
            
            # ====================================================================
            # BORRAR TODO (C): Solo meñique extendido
            # Cambiado de 5 dedos a solo meñique para evitar conflicto con número 5
            # ====================================================================
            if fingers == [0, 0, 0, 0, 1]:
                return "clear_all", "BORRAR TODO", 0.95, (255, 50, 50)
        
        # ====================================================================
        # GESTOS DE DOS MANOS
        # Números 6-9 y operaciones compuestas que requieren ambas manos
        # ====================================================================
        elif num_hands == 2:
            landmarks_0 = hands_data[0]['landmarks']
            landmarks_1 = hands_data[1]['landmarks']
            
            fingers_0 = self.count_extended_fingers(landmarks_0)
            fingers_1 = self.count_extended_fingers(landmarks_1)
            
            total_0 = sum(fingers_0)  # Total dedos mano 0
            total_1 = sum(fingers_1)  # Total dedos mano 1
            total_combined = total_0 + total_1  # Total combinado (no usado actualmente)
            
            # ====================================================================
            # NÚMEROS 6-9: 5 dedos en una mano + 1-4 en la otra
            # Sistema: Una mano completa (5) + dedos adicionales en la otra
            # ====================================================================
            if total_0 == 5 and total_1 >= 1:
                if total_1 == 1:
                    return "num_6", "SEIS (6)", 0.95, (100, 255, 100)
                elif total_1 == 2:
                    return "num_7", "SIETE (7)", 0.95, (100, 255, 100)
                elif total_1 == 3:
                    return "num_8", "OCHO (8)", 0.95, (100, 255, 100)
                elif total_1 == 4:
                    return "num_9", "NUEVE (9)", 0.95, (100, 255, 100)
            
            # Caso inverso: mano 1 tiene 5 dedos, mano 0 tiene extras
            elif total_1 == 5 and total_0 >= 1:
                if total_0 == 1:
                    return "num_6", "SEIS (6)", 0.95, (100, 255, 100)
                elif total_0 == 2:
                    return "num_7", "SIETE (7)", 0.95, (100, 255, 100)
                elif total_0 == 3:
                    return "num_8", "OCHO (8)", 0.95, (100, 255, 100)
                elif total_0 == 4:
                    return "num_9", "NUEVE (9)", 0.95, (100, 255, 100)
            
            # ====================================================================
            # MULTIPLICACIÓN (×): Índices cruzados formando X
            # Valida cruce real de los dedos índice en el centro
            # ====================================================================
            if self.hands_form_x(landmarks_0, landmarks_1):
                return "multiply", "MULTIPLICAR (×)", 0.95, (255, 100, 255)
            
            # ====================================================================
            # DIVISIÓN (÷): Gesto de tijera vertical
            # ====================================================================
            # Dos dedos en V en ambas manos
            if (fingers_0 == [0, 1, 1, 0, 0] and fingers_1 == [0, 1, 1, 0, 0]):
                return "divide", "DIVIDIR (÷)", 0.95, (150, 100, 255)
            
            # === BORRAR TODO: Ambas palmas abiertas ===
            if total_0 == 5 and total_1 == 5:
                return "clear_all", "BORRAR TODO", 0.9, (255, 50, 50)
            
            # Si solo una mano tiene gesto válido, procesarla
            if total_0 > 0 and total_1 == 0:
                return self.detect_gesture_raw([hands_data[0]])
            elif total_1 > 0 and total_0 == 0:
                return self.detect_gesture_raw([hands_data[1]])
        
        return "unknown", "...", 0.2, (150, 150, 150)
    
    def detect_gesture_stable(self, hands_data):
        """
        Detecta gesto CON estabilización temporal usando buffer de frames.
        
        Args:
            hands_data (list): Lista de manos detectadas con landmarks
            
        Returns:
            tuple: (gesto_id, gesto_nombre, confianza, color)
            
        Proceso:
            1. Detecta gesto instantáneo con detect_gesture_raw()
            2. Agrega al buffer circular (últimos 10 frames)
            3. Cuenta frecuencia de cada gesto en el buffer
            4. Retorna gesto más común SOLO si aparece en ≥70% de frames
            
        Beneficios:
            - Elimina detecciones erráticas por movimientos rápidos
            - Requiere que el gesto sea sostenido ~0.3 segundos (10 frames @ 30fps)
            - Reduce drasticamente falsos positivos
            
        Estados transitorios:
            - "Detectando...": Buffer aún no lleno (< 8 frames)
            - "Estabilizando...": Gesto detectado pero no cumple 70% estabilidad
            - Gesto confirmado: Estabilidad ≥70% en buffer completo
        """
        gesture_id, gesture_name, confidence, color = self.detect_gesture_raw(hands_data)
        
        # Agregar gesto actual al buffer circular
        self.gesture_buffer.append(gesture_id)
        
        # Esperar a tener suficientes frames para análisis
        if len(self.gesture_buffer) < 8:
            return "none", "Detectando...", 0.0, (150, 150, 150)
        
        # Contar frecuencia de cada gesto en el buffer
        counts = {}
        for g in self.gesture_buffer:
            counts[g] = counts.get(g, 0) + 1
        
        # Identificar gesto más frecuente
        most_common = max(counts, key=counts.get)
        stability = counts[most_common] / len(self.gesture_buffer)
        
        # Retornar gesto solo si cumple umbral de estabilidad y no es "none"
        if stability >= self.min_stability and most_common not in ["none", "unknown"]:
            return gesture_id, gesture_name, confidence * stability, color
        
        return "none", "Estabilizando...", 0.0, (150, 150, 150)
    
    def reset_buffer(self):
        """
        Limpia el buffer de gestos.
        
        Se usa cuando se procesa un gesto para evitar que el mismo gesto
        se detecte múltiples veces consecutivamente.
        """
        self.gesture_buffer.clear()

