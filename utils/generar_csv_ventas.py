import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generar_ventas_csv(nombre_archivo='ventas.csv', num_registros=100):
    """
    Genera un archivo CSV con datos de ventas ficticios.

    Args:
        nombre_archivo (str): El nombre del archivo CSV a crear.
        num_registros (int): El número de registros de ventas a generar.
    """
    print(f"Generando {num_registros} registros de ventas en '{nombre_archivo}'...")

    # Listas de datos posibles
    productos = [
        "Laptop", "Teclado", "Ratón", "Monitor", "Webcam", "Auriculares",
        "Smartphone", "Tableta", "Smartwatch", "Funda móvil", "Cargador",
        "Impresora", "Tinta", "Papel", "Disco Duro Externo", "Memoria USB",
        "Router", "Cable HDMI", "Altavoz Bluetooth", "Cámara digital"
    ]
    categorias = [
        "Electrónica", "Informática", "Accesorios", "Telefonía", "Periféricos", "Oficina"
    ]

    fechas = [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(num_registros)]
    precios = np.random.uniform(10, 1500, num_registros).round(2)
    cantidades = np.random.randint(1, 10, num_registros)

    # Crear los datos
    data = {
        'id': range(1, num_registros + 1),
        'producto': np.random.choice(productos, num_registros),
        'categoria': np.random.choice(categorias, num_registros),
        'fecha': [f.strftime('%Y-%m-%d') for f in fechas], # Formato YYYY-MM-DD
        'precio': precios,
        'cantidad': cantidades
    }

    df = pd.DataFrame(data)

    # Guardar en CSV
    df.to_csv(nombre_archivo, index=False)
    print(f"Archivo '{nombre_archivo}' creado con éxito.")

if __name__ == "__main__":
    generar_ventas_csv('./assets/ventas.csv', 100)