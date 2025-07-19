import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# --- 1. Carga de Datos ---
# Asegúrate de que el archivo 'reviews.csv' esté en la carpeta './sucio/'
archivo = "./assets/reviews.csv"
try:
    data = pd.read_csv(archivo)
    print("Datos cargados correctamente:")
    print(data.head())
    print("-" * 50)
except FileNotFoundError:
    print(f"Error: El archivo '{archivo}' no se encontró.")
    print("Asegúrate de que 'reviews.csv' esté en la rutra correcta.")
    exit()

# --- 2. Preprocesamiento: Vectorización de Texto ---
# Usamos TfidfVectorizer con stop words en inglés y n-gramas de 1 y 2 palabras
# Esto ayuda a capturar la importancia de las palabras y frases cortas como "no bueno"
vectorizador = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

# Transforma las reseñas en una matriz de características TF-IDF
X = vectorizador.fit_transform(data['review'])
# Las etiquetas de sentimiento (positivo/negativo)
y = data['sentiment']

print(f"Número total de características (palabras/n-gramas): {len(vectorizador.get_feature_names_out())}")
print("-" * 50)

# --- 3. División del Conjunto de Datos ---
# Dividimos los datos en conjuntos de entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
print("-" * 50)

# --- 4. Entrenamiento y Evaluación de Modelos ---

# Diccionario para almacenar los modelos para una fácil iteración
modelos = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Red Neuronal (MLP)": MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42)
}

# Diccionarios para almacenar métricas para la visualización
puntajes_exactitud = {}
matrices_confusion = {}

for nombre_modelo, modelo in modelos.items():
    print(f"\n--- Evaluando Modelo: {nombre_modelo} ---")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    exactitud = accuracy_score(y_test, y_pred)
    puntajes_exactitud[nombre_modelo] = exactitud

    cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)
    matrices_confusion[nombre_modelo] = cm

    print(f"Exactitud: {exactitud * 100:.2f}%")
    print("Informe de Clasificación:")
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusión:")
    print(cm)
    print("=" * 70)

# --- 5. Comparación Visual de Modelos ---

# --- a) Gráfico de Barras de Exactitud ---
plt.figure(figsize=(10, 6))
sns.barplot(x=list(puntajes_exactitud.keys()), y=list(puntajes_exactitud.values()), palette='viridis')
plt.title('Comparación de Exactitud de Modelos para Clasificación de Sentimientos')
plt.xlabel('Modelo')
plt.ylabel('Exactitud')
plt.ylim(0, 1) # La exactitud va de 0 a 1
for index, value in enumerate(puntajes_exactitud.values()):
    plt.text(index, value + 0.02, f'{value*100:.2f}%', ha='center', va='bottom')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('comparacion_exactitud_modelos.png') # Guarda la gráfica como imagen
plt.show() # Muestra la gráfica

# --- b) Mapas de Calor de Matrices de Confusión ---
# Ajusta dinámicamente el tamaño de la figura según el número de modelos
num_modelos = len(modelos)
fig, axes = plt.subplots(1, num_modelos, figsize=(6 * num_modelos, 6), sharey=True)

# Asegura que 'axes' sea una lista incluso para un solo modelo para evitar errores de indexación
if num_modelos == 1:
    axes = [axes]

for i, (nombre_modelo, cm) in enumerate(matrices_confusion.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=modelos[nombre_modelo].classes_,
                yticklabels=modelos[nombre_modelo].classes_)
    axes[i].set_title(f'Matriz de Confusión: {nombre_modelo}')
    axes[i].set_xlabel('Predicción')
    axes[i].set_ylabel('Real' if i == 0 else '') # Solo el primer gráfico tendrá la etiqueta Y

plt.tight_layout()
plt.savefig('comparacion_matrices_confusion.png') # Guarda todas las matrices en una imagen
plt.show()

# --- 6. Predicción con Nuevas Reseñas ---
print("\n--- Predicciones en Nuevas Reseñas ---")
nuevas_reseñas = [
    "La calidad es fantástica, me ha encantado.",
    "No me ha gustado para nada, una experiencia terrible.",
    "Muy mal servicio, no lo recomiendo en absoluto.",
    "El producto es bueno pero el envío fue lento.",
    "Absolutamente perfecto, superó mis expectativas.",
    "Una decepción total, no volveré a comprar.",
    "Funciona como se esperaba, bastante satisfecho.",
    "Increíblemente inútil, un desperdicio de dinero."
]

# Transforma las nuevas reseñas usando el mismo vectorizador
X_nuevas = vectorizador.transform(nuevas_reseñas)

print("\nResultados de predicción por modelo:")
print("-" * 40)

for nombre_modelo, modelo in modelos.items():
    predicciones = modelo.predict(X_nuevas)
    print(f"\n--- {nombre_modelo} ---")
    for reseña, sentimiento in zip(nuevas_reseñas, predicciones):
        print(f"  Reseña: \"{reseña}\" \n  Sentimiento: {sentimiento}\n")
    print("-" * 40)