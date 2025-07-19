import numpy as np
import pandas as pd

np.random.seed(42)
num_alumnos = 1000

df = pd.DataFrame({
    "calificacion_promedio": np.round(np.random.uniform(3, 10, num_alumnos), 2),
    "asistencia": np.round(np.random.uniform(50, 100, num_alumnos), 2),
    "horas_estudio": np.random.randint(0, 6, num_alumnos),
    "nivel_socioeconomico": np.random.choice([1, 2, 3], num_alumnos),
    "actividades_extracurriculares": np.random.choice([0, 1], num_alumnos),
    "problemas_conducta": np.random.choice([0, 1], num_alumnos)
})

# Generar probabilidad de abandono con una combinación ponderada
df["prob_abandono"] = (
    (10 - df["calificacion_promedio"]) * 0.3 +
    (100 - df["asistencia"]) * 0.2 +
    (5 - df["horas_estudio"]) * 0.2 +
    (df["nivel_socioeconomico"].map({1: 0.3, 2: 0.1, 3: 0})) +
    (df["problemas_conducta"] * 0.4) -
    (df["actividades_extracurriculares"] * 0.2)
)

# Normalizar entre 0 y 1
df["prob_abandono"] = (df["prob_abandono"] - df["prob_abandono"].min()) / (df["prob_abandono"].max() - df["prob_abandono"].min())

# Simular abandono como variable binaria con esa probabilidad
df["abandono"] = np.random.binomial(1, df["prob_abandono"])

# Eliminar columna auxiliar
df.drop(columns=["prob_abandono"], inplace=True)

# Guardar a CSV
df.to_csv("./assets/abandono_escolar.csv", index=False)
print("Datos generados y guardados correctamente.")
