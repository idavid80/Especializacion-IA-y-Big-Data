import pandas as pd
import random
from datetime import datetime, timedelta

# Función para generar una fecha aleatoria
def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

# Crear clientes
clientes = []
for i in range(50):
    fecha_nacimiento = random_date(datetime(1950, 1, 1), datetime(2005, 12, 31))
    cliente_id = f"C{i+1:03d}"
    clientes.append({'cliente_id': cliente_id, 'fecha_nacimiento': fecha_nacimiento})

# Generar 200 compras aleatorias
compras = []
for i in range(200):
    cliente = random.choice(clientes)
    fecha_compra = random_date(datetime(2022, 1, 1), datetime(2025, 7, 1))
    importe = round(random.uniform(10.0, 500.0), 2)
    compras.append({
        'cliente_id': cliente['cliente_id'],
        'fecha_compra': fecha_compra,
        'importe': importe,
        'fecha_nacimiento': cliente['fecha_nacimiento']
    })

# DataFrame de compras
df = pd.DataFrame(compras)

# Calcular edad
today = datetime.today()
df['edad'] = df['fecha_nacimiento'].apply(lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day)))

# Calcular compra promedio por cliente
resumen = df.groupby(['cliente_id', 'edad']).agg(
    compra_promedio=('importe', 'mean'),
    total_compras=('importe', 'count'),
    primera_compra=('fecha_compra', 'min'),
    ultima_compra=('fecha_compra', 'max')
).reset_index()

# Calcular años de actividad (mínimo 1 año para evitar división por cero)
resumen['años_actividad'] = ((resumen['ultima_compra'] - resumen['primera_compra']).dt.days / 365).clip(lower=1)

# Calcular compras anuales
resumen['compras_anuales'] = resumen['total_compras'] / resumen['años_actividad']

# Redondear columnas
resumen['compra_promedio'] = resumen['compra_promedio'].round(2)
resumen['compras_anuales'] = resumen['compras_anuales'].round(2)

# Dejar solo columnas finales
resumen_final = resumen[['cliente_id', 'edad', 'compra_promedio', 'compras_anuales']]

# Exportar a CSV
resumen_final.to_csv('./assets/clientes.csv', index=False)

print("Archivo 'clientes.csv' generado con compras anuales.")
