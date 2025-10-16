"""
Lógica de calculadora aritmética básica.

Este módulo contiene la clase Calculator que gestiona la construcción
incremental de expresiones matemáticas y su evaluación.
"""


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
            
        Uso:
            Mostrar en display secundario para feedback visual
        """
        expr = self.expression
        if self.current_number:
            expr += self.current_number
        return expr
