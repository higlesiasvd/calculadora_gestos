"""
Configuración de opciones de accesibilidad para usuarios.

Este módulo contiene la configuración centralizada para adaptar la calculadora
a diferentes necesidades de accesibilidad.
"""

# ============================================================================
# CLASE: AccessibilityConfig
# Propósito: Configuración de opciones de accesibilidad para usuarios
# Responsabilidades:
#   - Almacenar preferencias de voz (volumen, tono, velocidad, idioma)
#   - Gestionar modo de gestos extendidos (para movilidad reducida)
#   - Configurar umbrales de distancia y tiempo según necesidades
# ============================================================================
class AccessibilityConfig:
    """
    Configuración de accesibilidad para adaptar la calculadora a diferentes necesidades.
    
    Opciones disponibles:
        - Feedback por voz configurable (volumen, tono, velocidad, idioma)
        - Modo gestos extendidos (gestos más lentos y amplios)
        - Ayudas visuales dinámicas (guías de posicionamiento)
    """
    
    def __init__(self):
        """Inicializa configuración con valores por defecto."""
        # ====================================================================
        # CONFIGURACIÓN DE VOZ
        # ====================================================================
        self.voice_enabled = True           # Activar/desactivar feedback por voz
        self.voice_volume = 0.8             # Volumen (0.0-1.0)
        self.voice_rate = 150               # Velocidad de habla (palabras por minuto)
        self.voice_language = 'es'          # Idioma ('es', 'en', etc.)
        
        # ====================================================================
        # MODO GESTOS EXTENDIDOS (para usuarios con movilidad reducida)
        # ====================================================================
        self.extended_gestures = False      # Activar modo gestos extendidos
        self.gesture_hold_time = 15         # Frames para confirmar (normal: 15)
        self.extended_hold_time = 25        # Frames en modo extendido (más tiempo)
        self.distance_multiplier = 1.5      # Multiplicador de tolerancia de distancias
        
        # ====================================================================
        # AYUDAS VISUALES
        # ====================================================================
        self.show_hand_guides = True        # Mostrar guías de posicionamiento
        self.show_feedback_overlay = True   # Mostrar overlay de feedback
        self.guide_opacity = 0.6            # Opacidad de las guías (0.0-1.0)
    
    def get_hold_time(self):
        """Retorna el tiempo de hold según el modo activo."""
        return self.extended_hold_time if self.extended_gestures else self.gesture_hold_time
    
    def get_distance_threshold(self, base_distance):
        """
        Calcula el umbral de distancia según el modo.
        
        Args:
            base_distance (float): Distancia base en píxeles
            
        Returns:
            float: Distancia ajustada según multiplicador
        """
        return base_distance * (self.distance_multiplier if self.extended_gestures else 1.0)
