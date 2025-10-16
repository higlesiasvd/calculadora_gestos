"""
Sistema de feedback por voz usando pyttsx3.

Este módulo proporciona síntesis de voz para feedback auditivo,
ejecutándose de forma asíncrona para no bloquear la interfaz.
"""

import threading
import pyttsx3
from collections import deque


# ============================================================================
# CLASE: VoiceFeedback
# Propósito: Síntesis de voz para feedback auditivo
# Responsabilidades:
#   - Sintetizar texto a voz en español
#   - Ejecutar en hilo separado para no bloquear UI
#   - Gestionar cola de mensajes para evitar solapamiento
# ============================================================================
class VoiceFeedback:
    """
    Sistema de feedback por voz usando pyttsx3.
    
    Características:
        - Ejecución asíncrona (no bloquea la aplicación)
        - Cola de mensajes (un mensaje a la vez)
        - Configuración de volumen, tono y velocidad
        - Soporte multiidioma
    """
    
    def __init__(self, config):
        """
        Inicializa el motor de síntesis de voz.
        
        Args:
            config (AccessibilityConfig): Configuración de accesibilidad
        """
        self.config = config
        self.engine = None
        self.is_speaking = False
        self.message_queue = deque(maxlen=5)  # Cola de máximo 5 mensajes
        
        try:
            # Inicializar motor de síntesis de voz
            self.engine = pyttsx3.init()
            self._configure_engine()
            print("✓ Sistema de voz inicializado correctamente")
        except Exception as e:
            print(f"⚠ Advertencia: No se pudo inicializar el sistema de voz: {e}")
            self.config.voice_enabled = False
    
    def _configure_engine(self):
        """
        Configura el motor de voz con las preferencias del usuario.
        Busca automáticamente voces en español disponibles en el sistema.
        PRIORIDAD: Voces naturales de Apple (Monica, Paulina) > Voces Eloquence
        """
        if not self.engine:
            return
        
        try:
            # Configurar volumen (0.0 a 1.0)
            self.engine.setProperty('volume', self.config.voice_volume)
            
            # Configurar velocidad (palabras por minuto)
            self.engine.setProperty('rate', self.config.voice_rate)
            
            # Buscar voz en español (PRIORIDAD: voces naturales)
            voices = self.engine.getProperty('voices')
            spanish_voice_found = False
            
            # PRIORIDAD 1: Voces naturales de Apple (mejor calidad)
            natural_voices = ['monica', 'paulina', 'jorge', 'juan', 'diego']
            
            for voice in voices:
                voice_id_lower = voice.id.lower()
                voice_name_lower = voice.name.lower()
                
                # Buscar voces naturales primero (compact = voces naturales)
                for natural_name in natural_voices:
                    if (natural_name in voice_name_lower and 
                        'compact' in voice_id_lower):  # Voces compact son las naturales
                        self.engine.setProperty('voice', voice.id)
                        spanish_voice_found = True
                        print(f"✓ Voz natural en español: {voice.name}")
                        break
                
                if spanish_voice_found:
                    break
            
            # PRIORIDAD 2: Si no hay voces naturales, usar Eloquence
            if not spanish_voice_found:
                eloquence_names = ['eddy', 'flo', 'reed', 'sandy', 'shelley']
                
                for voice in voices:
                    voice_id_lower = voice.id.lower()
                    voice_name_lower = voice.name.lower()
                    
                    for eloquence_name in eloquence_names:
                        if (eloquence_name in voice_name_lower and 
                            'es-' in voice_id_lower):  # es-ES o es-MX
                            self.engine.setProperty('voice', voice.id)
                            spanish_voice_found = True
                            print(f"✓ Voz Eloquence en español: {voice.name}")
                            break
                    
                    if spanish_voice_found:
                        break
            
            if not spanish_voice_found:
                print("⚠ No se encontró voz en español. Usando voz predeterminada.")
                print("💡 Descarga voces desde: Configuración > Accesibilidad > Contenido hablado")
                    
        except Exception as e:
            print(f"⚠ Error al configurar voz: {e}")
    
    def speak(self, text):
        """
        Reproduce un mensaje de voz de forma asíncrona.
        
        Args:
            text (str): Texto a sintetizar
            
        Ejecución:
            - Si no hay mensajes pendientes: Reproduce inmediatamente
            - Si hay mensajes: Añade a la cola (solo si no está llena)
        """
        if not self.config.voice_enabled or not self.engine:
            return
        
        # Añadir a la cola y procesar en hilo separado
        self.message_queue.append(text)
        
        if not self.is_speaking:
            thread = threading.Thread(target=self._process_queue, daemon=True)
            thread.start()
    
    def _process_queue(self):
        """Procesa la cola de mensajes uno por uno."""
        self.is_speaking = True
        
        while len(self.message_queue) > 0:
            message = self.message_queue.popleft()
            try:
                self.engine.say(message)
                self.engine.runAndWait()
            except Exception as e:
                print(f"⚠ Error al reproducir voz: {e}")
        
        self.is_speaking = False
    
    def speak_number(self, number):
        """
        Reproduce un número en español de forma natural.
        
        Args:
            number (int): Número del 0 al 9 a pronunciar
        """
        numbers_es = {
            0: "cero", 1: "uno", 2: "dos", 3: "tres", 4: "cuatro",
            5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve"
        }
        self.speak(numbers_es.get(number, str(number)))
    
    def speak_operation(self, operation):
        """
        Reproduce el nombre de una operación matemática en español.
        
        Args:
            operation (str): Operador matemático (+, -, *, /)
        """
        operations_es = {
            "+": "más",
            "-": "menos",
            "*": "por",
            "/": "dividido"
        }
        self.speak(operations_es.get(operation, operation))
    
    def speak_result(self, result):
        """
        Reproduce el resultado de un cálculo de forma natural.
        
        Args:
            result: Resultado del cálculo (número o cadena)
        """
        # Convertir resultado a texto apropiado
        if isinstance(result, float):
            # Si es decimal, pronunciar con coma
            result_text = str(result).replace('.', ' coma ')
        else:
            result_text = str(result)
        
        self.speak(f"igual a {result_text}")
