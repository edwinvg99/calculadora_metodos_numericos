"""
?PARTICIPANTES:
 * Lina Verlliret Castrillon Zapata.
 * Mariana Ospina Mira.
 * Edwin Velasquez Giraldo.
"""

import sympy as sp
import numpy as np
from sympy import symbols, SympifyError, lambdify, sympify  

def newton_raphson(func_expr, deriv_expr, valor_inicial, margen_error, max_iteraciones):
    
    """
    newton-raphson:	
	1 columna: valor iniciar
	2 calcular function de valor iniciar
	3 ingresar function de valor inicar derivada
	4 aplicar metoodo:
		Valor inicial - function de valor iniciar / derivada 
	5 calcular error de aproximacion con:
		ABS((valor iniciar-resultado metodo)/ resultado metodo)*10
	6: nuevo valor iniciar sera = a resultado metodo punto 4
	7 bucle hasta el error deseado
    """
    
    # Definir la variable simbólica x
    x = sp.Symbol('x')
    
    # Convertir las expresiones simbólicas en funciones evaluables
    f = sp.lambdify(x, func_expr)  # Función original
    df = sp.lambdify(x, deriv_expr)  # Derivada de la función
    
    # Encabezado de la tabla de iteraciones
    print("{:<10} {:<20} {:<20} {:<20} {:<20}".format("Iter", "Valor Inicial", "f(Valor Inicial)", "f'(Valor Inicial)", "Error"))
    print("-" * 90)
    
    for i in range(max_iteraciones):
        # Evaluar la función y su derivada en el valor inicial
        funcion_valor_inicial = f(valor_inicial)
        derivada_valor_inicial = df(valor_inicial)
        
        # Si la derivada es 0, el método falla (evita división por cero)
        if derivada_valor_inicial == 0:
            print("Derivada cero, el método falla.")
            return None
        
        # Aplicar la fórmula del método de Newton-Raphson
        resultado_metodo = valor_inicial - funcion_valor_inicial / derivada_valor_inicial
        
        # Calcular el error de aproximación
        error_aproximacion = abs((valor_inicial - resultado_metodo) / resultado_metodo) * 100 if resultado_metodo != 0 else float('inf')
        
        # Imprimir los valores de la iteración actual
        print("{:<10} {:<20.10f} {:<20.10f} {:<20.10f} {:<20.10f}".format(i + 1, valor_inicial, funcion_valor_inicial, derivada_valor_inicial, error_aproximacion))
        
        # Verificar si se ha alcanzado la precisión deseada
        if error_aproximacion < margen_error:
            print("Convergencia alcanzada.")
            return resultado_metodo
        
        # Actualizar el valor inicial para la siguiente iteración
        valor_inicial = resultado_metodo
    
    # Si se alcanza el número máximo de iteraciones, devolver el último valor calculado
    print("Máximo de iteraciones alcanzado.")
    return valor_inicial

def biseccion():
    """
    Método de Bisección para encontrar una raíz de una función en un intervalo dado.
    
    Pasos:
    1. Se pide al usuario ingresar la función f(x).
    2. Se solicita el intervalo [a, b] donde se buscará la raíz.
    3. Se define la tolerancia para detener el proceso.
    4. Se verifica que el intervalo contenga una raíz (f(a) * f(b) < 0).
    5. Se aplica iterativamente el método de bisección hasta alcanzar la tolerancia deseada.
    6. Se imprime la raíz encontrada y el error relativo.
    
    Entradas:
    - Función f(x) ingresada por el usuario.
    - Intervalo [a, b] ingresado por el usuario.
    - Tolerancia ingresada por el usuario.
    
    Salida:
    - La raíz aproximada de la función dentro del intervalo dado.
    - Tabla de iteraciones con valores intermedios.
    """
    x = symbols('x')  # Definir la variable simbólica x
    fn = sympify(input("Ingresa la función: "))  # Convertir la entrada en una función simbólica
    f = lambdify(x, fn)  # Convertir la función simbólica en una función evaluable
    
    # Solicitar valores de a, b y la tolerancia
    a = float(input("Ingresa el valor de a: "))
    b = float(input("Ingresa el valor de b: "))
    tolerancia = float(input("Ingresa el valor de tolerancia: "))
    
    i = 0  # Contador de iteraciones
    error = 1  # Inicializar el error con un valor grande
    
    # Verificar si el intervalo [a, b] es válido (debe contener un cambio de signo)
    if f(a) * f(b) < 0:
        print("\n" + "{:^90}".format("MÉTODO DE BISECCIÓN"))
        print("{:^10} {:^10} {:^10} {:^10} {:^12} {:^12} {:^12} {:^10}".format(
            "i", "a", "b", "f(a)", "f(b)", "fa*fb", "error(%)", "p_medio"))
        
        # Aplicar el método de bisección iterativamente hasta alcanzar la tolerancia
        while error > tolerancia:
            p_medio = (a + b) / 2  # Calcular el punto medio
            fa = f(a)  # Evaluar la función en a
            fb = f(b)  # Evaluar la función en b
            fpm = f(p_medio)  # Evaluar la función en el punto medio
            fab = fa * fb  # Producto de f(a) y f(b), solo para referencia
            
            # Calcular el error relativo después de la primera iteración
            if i > 0:
                error = abs((b - a) / 2) * 100
            
            # Imprimir la iteración actual
            print(f"{i:^10} {a:^10.5f} {b:^10.5f}  {fa:^12.5f} {fb:^12.5f} {fab:^12.5f} {error:^10.5f} {p_medio:^10.5f}")
            
            # Decidir el nuevo intervalo en función del signo de f(p_medio)
            if fa * fpm < 0:
                b = p_medio  # La raíz está en [a, p_medio]
            else:
                a = p_medio  # La raíz está en [p_medio, b]
            
            i += 1  # Incrementar el contador de iteraciones
        
        # Imprimir el resultado final
        print(f"\nLa raíz es: {p_medio:.5f} con un error de {error:.5f}%")
    else:
        print("Esta función no cruza el eje X en el intervalo dado, por lo que no hay una raíz real en este rango.")

def reglaFalsa():
    """
    Implementa el método de la Regla Falsa para encontrar una raíz de una función dentro de un intervalo dado.
    
    Pasos del algoritmo:
    1. Se solicita al usuario ingresar la función f(x), los valores del intervalo [a, b] y la tolerancia.
    2. Se evalúan los valores de f(a) y f(b) para verificar si hay un cambio de signo (condición para la existencia de raíz en el intervalo).
    3. Se calcula la aproximación de la raíz usando la fórmula de la Regla Falsa.
    4. Se itera hasta que el error sea menor que la tolerancia especificada.
    5. Se imprime la tabla de iteraciones y el valor final de la raíz encontrada.
    
    Restricciones:
    - La función debe ser continua en el intervalo [a, b].
    - Debe existir un cambio de signo en f(a) y f(b), es decir, f(a) * f(b) < 0.
    """
    
    # Definir la variable simbólica y la función
    x = symbols('x')
    fn = sympify(input("Ingresa la función: "))  # Función a utilizar
    f = lambdify(x, fn)
    # Iniciar variables
    a = float(input("Ingresa el valor de a: "))
    b = float(input("Ingresa el valor de b: "))
    tolerancia = float(input("Ingresa el valor de tolerancia: "))
    
    # Preguntar al usuario si desea limitar el número de iteraciones
    limitar_iteraciones = input("¿Desea limitar el número máximo de iteraciones? (s/n): ").lower()
    max_iteraciones = float('inf')  # Por defecto, sin límite
    
    if limitar_iteraciones == 's' or limitar_iteraciones == 'si':
        max_iteraciones = int(input("Ingrese el número máximo de iteraciones: "))
    
    i = 0  # Contador de iteraciones
    error = 1  # Inicializar el error
    p_anterior = 0  # Inicializar el valor anterior de p_medio
    # Calcular valores iniciales de la función en a y b
    fa = f(a)
    fb = f(b)
    # Verificar si hay un cambio de signo en el intervalo
    if fa * fb < 0:
        # Encabezado de la tabla con las nuevas columnas
        print("\n")
        print("{:^90}".format("MÉTODO DE LA REGLA FALSA"))
        print("{:^5} {:^10} {:^10} {:^10} {:^10} {:^12} {:^12} {:^15} {:^15}".format(
            "i", "a", "b", "f(a)", "f(b)", "p_medio", "f(p_medio)", "f(a)*f(m)", "f(b)*f(m)"))
        while error > tolerancia and i < max_iteraciones:
            # Calcular el nuevo punto medio con la Regla Falsa
            p_medio = (a * fb - b * fa) / (fb - fa)
            fpm = f(p_medio)
            # Calcular los productos f(a) * f(medio) y f(b) * f(medio)
            fapm = fa * fpm
            fbpm = fb * fpm
            # Calcular el error relativo (excepto en la primera iteración)
            if i > 0:
                error = abs((p_medio - p_anterior) / p_medio) * 100
            # Imprimir los valores de la iteración actual
            print(f"{i:^5} {a:^10.5f} {b:^10.5f}  {fa:^10.5f} {fb:^10.5f} {p_medio:^12.5f} {fpm:^12.5f} {fapm:^15.5f} {fbpm:^15.5f}")
            # Actualizar los valores de a o b según la Regla Falsa
            if fapm < 0:
                b = p_medio
                fb = fpm
            else:
                a = p_medio
                fa = fpm
            # Guardar el valor anterior de p_medio
            p_anterior = p_medio
            i += 1
        
        # Mostrar mensaje según cómo terminó el algoritmo
        if error <= tolerancia:
            print(f"\nLa raíz es: {p_medio:.5f} con un error de {error:.5f}%")
        else:
            print(f"\nSe alcanzó el número máximo de iteraciones ({max_iteraciones}).")
            print(f"La mejor aproximación encontrada es: {p_medio:.5f} con un error de {error:.5f}%")
    else:
        print("El intervalo no es válido para la Regla Falsa.")
def puntoFijo():
    """
    Metodo de Punto Fijo:
    1. Se ingresa la función original f(x) y la despejada g(x).
    2. Se verifica que |g'(x)| < 1 en el punto inicial para asegurar convergencia.
    3. Se calcula iterativamente x_{n+1} = g(x_n).
    4. Se mide el error relativo: ABS((x_n+1 - x_n) / x_n+1) * 100.
    5. Itera hasta que el error sea menor al margen dado o hasta alcanzar el maximo de iteraciones.
    6. Se evalua f(x) con la raíz obtenida para verificar si realmente es una solución.
    """

    # Definir la variable simbólica x
    x = sp.Symbol('x')

    # Solicitar la función original f(x) y la despejada g(x)
    f_expr = sp.sympify(input("Ingrese la función original f(x) en términos de x: "))
    g_expr = sp.sympify(input("Ingrese la función despejada g(x) en términos de x: "))

    # Convertir las expresiones en funciones evaluables
    f = sp.lambdify(x, f_expr)  # f(x)
    g = sp.lambdify(x, g_expr)  # g(x)
    g_derivada_expr = sp.diff(g_expr, x)  # g'(x)
    g_derivada = sp.lambdify(x, g_derivada_expr)  # g'(x) evaluable

    # Pedir parámetros al usuario
    valor_inicial = float(input("Ingrese el valor inicial: "))
    margen_error = float(input("Ingrese el margen de error (%): "))
    max_iteraciones = int(input("Ingrese el número máximo de iteraciones: "))

    # Verificar convergencia con |g'(x0)| < 1
    try:
        if abs(g_derivada(valor_inicial)) >= 1:
            print("Advertencia: La función g(x) podría no converger porque |g'(x0)| >= 1.")
            return None
    except:
        print("Error al evaluar la derivada de g(x). Verifique su despeje.")
        return None

    # Encabezado de la tabla de iteraciones
    print("\n               MÉTODO DE PUNTO FIJO")
    print("{:<10} {:<20} {:<20} {:<20}".format("Iter", "x_i", "g(x_i)", "Error (%)"))
    print("-" * 70)

    for i in range(max_iteraciones):
        # Calcular el siguiente valor usando g(x)
        nuevo_valor = g(valor_inicial)

        # Calcular el error relativo
        error_aproximacion = abs((nuevo_valor - valor_inicial) / nuevo_valor) * 100 if nuevo_valor != 0 else float('inf')

        # Imprimir los valores de la iteración actual
        print("{:<10} {:<20.10f} {:<20.10f} {:<20.10f}".format(i + 1, valor_inicial, nuevo_valor, error_aproximacion))

        # Verificar si se ha alcanzado la precisión deseada
        if error_aproximacion < margen_error:
            print("\nConvergencia alcanzada.")
            print(f"Evaluando f({nuevo_valor}): {f(nuevo_valor)}")
            if abs(f(nuevo_valor)) < margen_error:
                print("La raíz encontrada es válida.")
            else:
                print("Advertencia: La raíz encontrada no satisface f(x) ≈ 0.")
            return nuevo_valor

        # Actualizar el valor inicial para la siguiente iteración
        valor_inicial = nuevo_valor

    # Si se alcanza el número máximo de iteraciones
    print("\nMáximo de iteraciones alcanzado.")
    print(f"Evaluando f({valor_inicial}): {f(valor_inicial)}")
    if abs(f(valor_inicial)) < margen_error:
        print("La raíz encontrada es válida.")
    else:
        print("Advertencia: La raíz encontrada no satisface f(x) ≈ 0.")
    return valor_inicial

def secante(f, a, b, tol, max_iter=200000):
    """
    Método de la secante para encontrar la raíz de una función f(x).

    Parámetros:
    f : function
        Función que se desea encontrar su raíz.
    a : float
        Primer valor inicial del intervalo.
    b : float
        Segundo valor inicial del intervalo.
    tol : float
        Tolerancia para la convergencia del método.
    max_iter : int, opcional
        Número máximo de iteraciones permitidas (por defecto 100).

    Retorna:
    tuple : (float, int)
        Aproximación de la raíz y número de iteraciones realizadas.
    """
    
    iter_count = 0  # Contador de iteraciones
    print("\nIteración |     x_n     |    f(x_n)    |    Error    ")
    print("-------------------------------------------------")

    # Iteración del método de la secante
    while abs(b - a) > tol and iter_count < max_iter:
        try:
            # Cálculo del siguiente punto usando la fórmula de la secante
            x_new = b - (f(b) * (b - a)) / (f(b) - f(a))  
        except ZeroDivisionError:
            print("División por cero detectada, prueba con otros valores iniciales.")
            return None, iter_count

        error = abs(x_new - b)  # Cálculo del error relativo
        print(f"   {iter_count+1:2d}    | {b:10.6f} | {f(b):10.6f} | {error:10.6f} ")

        # Actualización de valores para la siguiente iteración
        a, b = b, x_new
        iter_count += 1

    print("-------------------------------------------------")

    # Retorna la raíz aproximada si convergió, de lo contrario None
    return b if iter_count < max_iter else None, iter_count


if __name__ == "__main__":
    while True:
        print("\nSeleccione el método:")
        print("1. Método de Newton-Raphson")
        print("2. Método de Bisección")
        print("3. Método de regla falsa")
        print("4. Método de punto fijo")
        print("5. Método de secante")
        print("0. Salir")
        opcion = input("Ingrese el número de la opción: ")
        
        # metodo newton-raphson
        if opcion == "1":

            # Solicitar la función al usuario como una entrada de texto
            expr = input("Ingrese la función en términos de x: ejemplo: x**3-10*x-5 ")
            
            # Definir la variable simbólica
            x = sp.Symbol('x')
            
            # Convertir la entrada en una expresión simbólica y calcular su derivada
            func_expr = sp.sympify(expr)
            deriv_expr = sp.diff(func_expr, x)
            
            # Pedir parámetros al usuario
            valor_inicial = float(input("Ingrese el valor inicial: "))
            margen_error = float(input("Ingrese el margen de error (%): "))
            max_iteraciones = int(input("Ingrese el número máximo de iteraciones: "))
            
            # Mostrar la función y su derivada
            print(f"Función: {func_expr}")
            print(f"Derivada: {deriv_expr}")
            
            # Llamar al método de Newton-Raphson
            raiz = newton_raphson(func_expr, deriv_expr, valor_inicial, margen_error, max_iteraciones)
            
            # se immprime la raiz aproximada si se encontro
            if raiz is not None:
                print(f"Raíz aproximada: {raiz}")
                
        elif opcion == "2":
            biseccion()
            
        elif opcion == "3":
            reglaFalsa()

        elif opcion == "4":
            puntoFijo()
    
        elif opcion == "5":
                    
            # Solicitar al usuario la función como string y convertirla a función con eval
            funcion = input("Ingresa la función en términos de x (ejemplo: np.sin(x) - x/2): ")
            funcionConvertida = lambda x: eval(funcion, {"x": x, "np": np})  # Evalúa la función ingresada

            # Pedir valores iniciales y tolerancia
            a = float(input("Ingrese el primer punto (x0): "))
            b = float(input("Ingrese el segundo punto (x1): "))
            tol = float(input("Ingrese la tolerancia: "))

            # Llamar al método de la secante
            raiz, iteraciones = secante(funcionConvertida, a, b, tol)

            # Mostrar resultado
            if raiz is not None:
                print(f"\nLa raíz aproximada es: {raiz:.4f} encontrada en {iteraciones} iteraciones.")
            else:
                print("\nNo se encontró una raíz en el número máximo de iteraciones.")
            
        elif opcion == "0":
            print("Saliendo del programa.")
            break
        
        else:
            print("Opción no válida. Intente de nuevo.")
