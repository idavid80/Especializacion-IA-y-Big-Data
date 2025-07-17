import pandas as pd
import sqlite3
import os

# Paso 1: Extracción de datos
def extraer_datos(filepath):
    """
    Carga datos desde un archivo CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo {filepath} no existe.")
    print("Extrayendo datos del archivo CSV...")
    datos = pd.read_csv(filepath)
    print("Datos extraídos con éxito.")
    return datos

# Paso 2: Transformación de datos
def transformar_datos(datos):
    """
    Limpia y transforma los datos.
    """
    print("Transformando datos...")
    datos['fecha'] = pd.to_datetime(datos['fecha'])  # Convertir fechas
    datos['total'] = datos['precio'] * datos['cantidad']  # Calcular total
    datos_transformados = datos[['id', 'producto', 'categoria', 'fecha', 'total']]
    print("Transformación completada.")
    return datos_transformados

# Paso 3: Carga de datos
def cargar_datos(datos, db_name='./soluciones/bigdata/lab1/ventas.db', table_name='ventas'):
    """
    Carga los datos transformados en una base de datos SQLite.
    """
    print(f"Cargando datos en la base de datos {db_name}...")
    conexion = sqlite3.connect(db_name)
    datos.to_sql(table_name, conexion, if_exists='replace', index=False)
    conexion.close()
    print("Datos cargados exitosamente.")

# Visualizar datos cargados en SQLite (MODIFICADA)
def visualizar_datos(db_name, query, headers=None):
    """
    Ejecuta una consulta SQL en la base de datos SQLite y muestra los resultados de forma dinámica.
    """
    conexion = sqlite3.connect(db_name)
    cursor = conexion.cursor()
    print(f"Ejecutando consulta: {query}\n")

    cursor.execute(query)
    resultados = cursor.fetchall()

    # Obtener nombres de las columnas de forma dinámica si no se proporcionan
    if headers is None:
        headers = [description[0] for description in cursor.description]

    # Imprimir cabecera
    header_line = ""
    separator_line = ""
    for header in headers:
        header_line += f"{header:<20}" # Aumentamos el ancho para mejor visualización
        separator_line += "-" * 20

    print(header_line)
    print(separator_line)

    # Mostrar resultados
    if not resultados:
        print("No se encontraron resultados.")
    else:
        for row in resultados:
            row_line = ""
            for i, item in enumerate(row):
                # Formatear el total como float con 2 decimales si es el último elemento y es numérico
                if headers[i].lower() == 'total' or headers[i].lower() == 'total_categoria':
                     row_line += f"{item:<20.2f}"
                else:
                    row_line += f"{str(item):<20}" # Convertir a string para asegurar el formateo
            print(row_line)
    print("\n") # Añadir un espacio extra al final
    conexion.close()

# Flujo principal del proceso ETL
def proceso_etl(filepath):
    """
    Ejecución completa del flujo ETL.
    """
    print("Iniciando el proceso ETL...")
    # Extracción
    datos = extraer_datos(filepath)
    print("\n--- Vista previa de datos extraídos ---")
    print(datos.head())

    # Transformación
    datos_transformados = transformar_datos(datos)
    print("\n--- Vista previa de datos transformados ---")
    print(datos_transformados.head())

    # Carga
    cargar_datos(datos_transformados)
    print("Proceso ETL finalizado con éxito.")

if __name__ == "__main__":
    # When running 'python3 ./ejercicios/etl.py' from the root of the repository,
    # the current working directory is the repository root.
    # So, the path to ventas.csv needs to be relative to that root.
    archivo_csv = './soluciones/assets/ventas.csv'

    try:
        proceso_etl(archivo_csv)

        # Ensure the database path is also correct relative to the execution directory
        # If the script creates 'ventas.db' inside 'soluciones/bigdata/lab1/',
        # then this path should also be relative to the root when calling visualize_datos
        db_name = './soluciones/bigdata/lab1/ventas.db'

        print("--- Todas las Ventas ---")
        mostrar_ventas = "SELECT id, producto, categoria, fecha, total FROM ventas;"
        visualizar_datos(db_name, mostrar_ventas)

        print("--- Total de Ventas por Categoría ---")
        total_categorias = "SELECT categoria, SUM(total) AS total_categoria FROM ventas GROUP BY categoria;"
        visualizar_datos(db_name, total_categorias)

        print("--- Ventas con Total Mayor a 100 ---")
        ventas_mayor_100 = "SELECT id, producto, categoria, fecha, total FROM ventas WHERE total > 100;"
        visualizar_datos(db_name, ventas_mayor_100)

        print("--- Ventas Ordenadas por Fecha (Ascendente) ---")
        ordenar_fechas = "SELECT id, producto, categoria, fecha, total FROM ventas ORDER BY fecha ASC;"
        visualizar_datos(db_name, ordenar_fechas)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Asegúrate de que 'ventas.csv' esté en la ubicación correcta.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

    input("Presiona Enter para salir...")