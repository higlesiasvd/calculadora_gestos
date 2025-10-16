# ============================================================================
# IMPORTS - Librerías necesarias para el funcionamiento de la aplicación
# ============================================================================
import cv2                      # OpenCV - Procesamiento de video y renderizado UI
import mediapipe as mp          # MediaPipe - Detección de landmarks de manos
import time                     # Control de tiempos y delays
import numpy as np              # Operaciones numéricas y vectoriales
from collections import deque   # Buffer circular para estabilización de gestos
import math                     # Operaciones matemáticas (ángulos, distancias)


# ============================================================================
# CLASE: GestureDetector
# Propósito: Detectar y reconocer gestos de manos usando MediaPipe
# Responsabilidades:
#   - Detectar landmarks de hasta 2 manos simultáneamente
#   - Identificar gestos basados en posición y orientación de dedos
#   - Estabilizar detecciones mediante buffer temporal
#   - Validar gestos con alta precisión (distancias, ángulos)
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
        Detecta si dos manos forman una cruz '+' (para suma) - MEJORADO con cálculo de ángulos.
        
        Args:
            landmarks_1 (list): Landmarks de la primera mano
            landmarks_2 (list): Landmarks de la segunda mano
            
        Returns:
            bool: True si las manos forman una cruz perpendicular
            
        Algoritmo mejorado:
            1. Usa dedo medio como referencia (más estable que índice)
            2. Calcula vectores desde muñeca hasta punta del medio
            3. Calcula ángulo real entre vectores usando producto punto
            4. Valida perpendicularidad (70°-110°) para evitar falsos positivos
            5. Verifica distancia mínima entre puntas (300px) para confirmar formación
            
        Ventajas sobre versión anterior:
            - No depende de orientación cardinal (horizontal/vertical estricto)
            - Acepta cruces con ligera rotación
            - Matemáticamente preciso (ángulo real, no aproximación)
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
        
        # Cruz si el ángulo está cerca de 90 grados (70-110 grados)
        # Verificar si las puntas están razonablemente cerca (cruz formada)
        distance = math.sqrt(
            (middle_1_tip['x'] - middle_2_tip['x'])**2 + 
            (middle_1_tip['y'] - middle_2_tip['y'])**2
        )
        
        # Cruz válida si: ángulo entre 70-110° Y distancia < 300px
        is_cross = 70 < angle < 110 and distance < 300
        
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
            # SUMA (+): Manos formando cruz perpendicular - MEJORADO
            # Ambas manos deben tener SOLO índice extendido para máxima precisión
            # Previene falsos positivos con otras configuraciones de dedos
            # ====================================================================
            index_only_0 = fingers_0 == [0, 1, 0, 0, 0]
            index_only_1 = fingers_1 == [0, 1, 0, 0, 0]
            
            if index_only_0 and index_only_1 and self.hands_form_cross(landmarks_0, landmarks_1):
                return "add", "+ SUMA", 0.98, (0, 255, 0)
            
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


# ============================================================================
# CLASE: Calculator
# Propósito: Lógica de calculadora aritmética básica
# Responsabilidades:
#   - Construir números dígito por dígito
#   - Construir expresiones matemáticas (ej: "5+3*2")
#   - Evaluar expresiones usando eval() de Python
#   - Gestionar estado (número actual, expresión, resultado)
# ============================================================================
class Calculator:
    """
    Lógica de calculadora aritmética con construcción incremental.
    
    Modelo de operación:
        1. Usuario ingresa dígitos → se acumulan en current_number
        2. Usuario selecciona operación → current_number se añade a expression
        3. Usuario repite hasta completar expresión (ej: "5+3*2")
        4. Usuario presiona = → se evalúa expression con eval()
        
    Variables de estado:
        - current_number: Dígitos del número actual siendo ingresado
        - expression: Expresión matemática completa (ej: "5+3*2")
        - result: Resultado del último cálculo
        - decimal_mode: Flag para manejo de decimales (no implementado totalmente)
    """
    
    def __init__(self):
        """Inicializa calculadora en estado vacío."""
        self.current_number = ""    # Número siendo construido actualmente
        self.expression = ""        # Expresión matemática completa
        self.result = ""            # Resultado del último cálculo
        self.decimal_mode = False   # Modo decimal (no usado actualmente)
    
    def add_digit(self, digit):
        """
        Añade un dígito al número actual.
        
        Args:
            digit (int): Dígito 0-9 a añadir
            
        Returns:
            bool: True si se añadió exitosamente, False si se alcanzó límite
            
        Límite: Máximo 12 dígitos para prevenir overflow visual
        """
        if len(self.current_number) < 12:
            self.current_number += str(digit)
            return True
        return False
    
    def add_decimal(self):
        """
        Añade punto decimal al número actual.
        
        Returns:
            bool: True si se añadió, False si ya existe punto decimal
            
        Comportamiento:
            - Si número está vacío: Añade "0."
            - Si número existe sin punto: Añade "." al final
            - Si ya tiene punto: Retorna False (un solo decimal permitido)
        """
        if "." not in self.current_number:
            if not self.current_number:
                self.current_number = "0."
            else:
                self.current_number += "."
            return True
        return False
    
    def add_operation(self, op):
        """
        Añade una operación matemática a la expresión.
        
        Args:
            op (str): Operador matemático ("+", "-", "*", "/")
            
        Returns:
            bool: True si se añadió exitosamente
            
        Comportamiento:
            1. Si hay resultado previo pero no número actual:
               Usa el resultado como primer operando (ej: "42 + ")
            2. Si hay número actual:
               Añade número a expresión con operador (ej: "5 + ")
            3. Si no hay nada: Retorna False
            
        Ejemplo de flujo:
            current_number="5" → add_operation("+") → expression="5 + "
            current_number="3" → add_operation("*") → expression="5 + 3 * "
        """
        # Caso 1: Reutilizar resultado previo como primer operando
        if not self.current_number and self.result:
            self.expression = self.result + " " + op + " "
            self.result = ""
            return True
        
        # Caso 2: Añadir número actual a la expresión
        if self.current_number:
            self.expression += self.current_number + " " + op + " "
            self.current_number = ""
            return True
        
        return False
    
    def calculate(self):
        """
        Evalúa la expresión matemática completa y retorna el resultado.
        
        Returns:
            tuple: (éxito: bool, resultado: str)
                - (True, "42"): Cálculo exitoso
                - (False, "Error"): Error en evaluación
                - (False, ""): No hay expresión para evaluar
                
        Proceso:
            1. Completa expresión con número actual si existe
            2. Evalúa usando eval() de Python (soporta +, -, *, /)
            3. Formatea resultado (enteros sin decimales, floats truncados)
            4. Limpia expresión y retorna resultado
            
        Formateo de resultados:
            - 42.0 → "42" (enteros sin decimal)
            - 3.14159265 → "3.141593" (máximo 6 decimales, sin ceros finales)
            - Error de sintaxis → "Error"
            
        Nota de seguridad:
            eval() es usado aquí en contexto controlado (solo expresiones numéricas),
            pero en producción se recomienda usar ast.literal_eval o un parser dedicado.
        """
        # Completar expresión con número actual si está pendiente
        if self.current_number:
            self.expression += self.current_number
            self.current_number = ""
        
        if self.expression:
            try:
                # EVALUAR EXPRESIÓN - eval() ejecuta la expresión matemática
                result = eval(self.expression)
                
                # Formatear resultado según tipo
                if isinstance(result, float):
                    if result.is_integer():
                        # Float que es entero (42.0) → mostrar sin decimales
                        self.result = str(int(result))
                    else:
                        # Float con decimales → truncar a 6 dígitos, quitar ceros finales
                        self.result = f"{result:.6f}".rstrip('0').rstrip('.')
                else:
                    self.result = str(result)
                
                self.expression = ""  # Limpiar expresión tras cálculo exitoso
                return True, self.result
            except:
                # Error en evaluación (división por 0, sintaxis inválida, etc.)
                self.result = "Error"
                self.expression = ""
                return False, "Error"
        return False, ""
    
    def clear_all(self):
        """
        Borra TODO el estado de la calculadora (C = Clear).
        
        Resetea:
            - current_number: Número siendo ingresado
            - expression: Expresión matemática
            - result: Resultado previo
            
        Equivalente al botón "C" o "AC" de calculadoras físicas.
        """
        self.current_number = ""
        self.expression = ""
        self.result = ""
    
    def backspace(self):
        """
        Borra el último carácter ingresado (← = Backspace).
        
        Comportamiento:
            - Si hay número actual: Borra último dígito
            - Si no hay número pero sí expresión: Borra último carácter de expresión
            
        Nota: Actualmente no tiene gesto asignado, pero puede agregarse.
        """
        if self.current_number:
            self.current_number = self.current_number[:-1]
        elif self.expression:
            self.expression = self.expression[:-1].rstrip()
    
    def get_display(self):
        """
        Obtiene el texto a mostrar en el display principal.
        
        Returns:
            str: Resultado si existe, número actual si existe, "0" si está vacío
            
        Prioridad:
            1. Resultado de cálculo previo
            2. Número actual siendo ingresado
            3. "0" por defecto
        """
        if self.result:
            return self.result
        return self.current_number if self.current_number else "0"
    
    def get_expression(self):
        """
        Obtiene la expresión completa incluyendo el número actual.
        
        Returns:
            str: Expresión matemática parcial o completa
            
        Ejemplo:
            expression="5 + 3 * ", current_number="2"
            → retorna "5 + 3 * 2"
        """
        return self.expression + self.current_number


# ============================================================================
# CLASE: UIRenderer
# Propósito: Renderizar interfaz gráfica de la calculadora sobre el video
# Responsabilidades:
#   - Dibujar display de calculadora con expresión y resultado
#   - Mostrar indicador de gesto actual detectado
#   - Dibujar guía lateral con lista de gestos disponibles
#   - Mostrar feedback temporal de acciones
#   - Renderizar barra de cooldown (tiempo de espera entre gestos)
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
    """
    
    def __init__(self, width, height):
        """
        Inicializa el renderizador con dimensiones de la ventana.
        
        Args:
            width (int): Ancho de la ventana en píxeles
            height (int): Alto de la ventana en píxeles
        """
        self.width = width
        self.height = height
        self.feedback_msg = ""               # Mensaje de feedback actual
        self.feedback_timer = 0              # Frames restantes para mostrar feedback
        self.feedback_color = (0, 255, 0)   # Color del feedback
    
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


# ============================================================================
# CLASE: GestureCalculatorApp
# Propósito: Aplicación principal que integra todos los componentes
# Responsabilidades:
#   - Gestionar captura de video desde cámara
#   - Coordinar detector de gestos, calculadora y renderizador UI
#   - Implementar sistema de hold-to-confirm (mantener gesto 0.5s)
#   - Gestionar cooldowns entre gestos
#   - Procesar gestos y actualizar estado de calculadora
#   - Renderizar frame completo y gestionar ciclo principal
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
    
    def __init__(self, camera_index=0):
        """
        Inicializa la aplicación y configura la cámara.
        
        Args:
            camera_index (int): Índice de la cámara (0 = cámara predeterminada)
            
        Configuración de cámara:
            - Resolución: 1920x1080 (Full HD)
            - FPS: 30 frames por segundo
            - Buffer: 1 frame (minimiza latencia)
            
        Raises:
            Exception: Si no se puede abrir la cámara
        """
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
        self.detector = GestureDetector()                  # Detector de gestos
        self.calc = Calculator()                           # Lógica de calculadora
        self.ui = UIRenderer(self.width, self.height)      # Renderizador UI
        
        # Variables de control de gestos
        self.last_gesture = "none"      # Último gesto procesado (previene duplicados)
        self.cooldown = 0               # Frames restantes de cooldown
        self.cooldown_time = 25         # Cooldown estándar (~0.8s @ 30fps)
        
        # Variables de FPS (frames por segundo)
        self.fps_time = time.time()
        self.fps = 0
    
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
                self.cooldown = self.cooldown_time
                self.detector.reset_buffer()
        
        # ====================================================================
        # SUMA (+): Añadir operador de suma
        # ====================================================================
        elif gid == "add":
            if self.calc.add_operation("+"):
                self.ui.show_feedback("+ SUMA", (0, 255, 0))
                self.cooldown = self.cooldown_time
                self.detector.reset_buffer()
        
        # ====================================================================
        # RESTA (-): Añadir operador de resta
        # ====================================================================
        elif gid == "subtract":
            if self.calc.add_operation("-"):
                self.ui.show_feedback("- RESTA", (255, 150, 0))
                self.cooldown = self.cooldown_time
                self.detector.reset_buffer()
        
        # ====================================================================
        # MULTIPLICACIÓN (×): Añadir operador de multiplicación
        # ====================================================================
        elif gid == "multiply":
            if self.calc.add_operation("*"):
                self.ui.show_feedback("x MULTIPLICAR", (255, 100, 255))
                self.cooldown = self.cooldown_time
                self.detector.reset_buffer()
        
        # ====================================================================
        # DIVISIÓN (÷): Añadir operador de división
        # ====================================================================
        elif gid == "divide":
            if self.calc.add_operation("/"):
                self.ui.show_feedback("/ DIVIDIR", (150, 100, 255))
                self.cooldown = self.cooldown_time
                self.detector.reset_buffer()
        
        # ====================================================================
        # IGUAL (=): Calcular resultado de la expresión
        # ====================================================================
        elif gid == "equal":
            success, result = self.calc.calculate()
            if success:
                self.ui.show_feedback(f"= {result}", (0, 255, 255), 60)
            else:
                self.ui.show_feedback("Error", (255, 50, 50))
            self.cooldown = self.cooldown_time
            self.detector.reset_buffer()
        
        # ====================================================================
        # BORRAR TODO (C): Resetear calculadora completamente
        # ====================================================================
        elif gid == "clear_all":
            self.calc.clear_all()
            self.ui.show_feedback("TODO BORRADO", (255, 50, 50))
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
        print("\nPresiona ESC o 'q' para salir\n")
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
            
            # ================================================================
            # CÁLCULO Y DISPLAY DE FPS
            # ================================================================
            current_time = time.time()
            self.fps = 1 / (current_time - self.fps_time + 1e-6)  # +epsilon evita div/0
            self.fps_time = current_time
            
            cv2.putText(frame, f"FPS: {int(self.fps)}", (self.width - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # ================================================================
            # INSTRUCCIONES DE SALIDA
            # ================================================================
            cv2.putText(frame, "Presiona ESC o 'q' para salir", 
                       (50, self.height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # ================================================================
            # MOSTRAR FRAME Y PROCESAR INPUT
            # ================================================================
            cv2.imshow('Calculadora Gestual', frame)
            
            # Esperar 1ms y procesar teclas presionadas
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC o 'q' para salir
                break
        
        # ====================================================================
        # LIMPIEZA Y CIERRE
        # ====================================================================
        self.cap.release()          # Liberar cámara
        cv2.destroyAllWindows()     # Cerrar ventanas de OpenCV
        print("\nOK Aplicacion cerrada correctamente")


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================
if __name__ == "__main__":
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
        - Cámara conectada (índice 0)
    """
    try:
        # Crear instancia de la aplicación (cámara índice 0)
        app = GestureCalculatorApp(camera_index=0)
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