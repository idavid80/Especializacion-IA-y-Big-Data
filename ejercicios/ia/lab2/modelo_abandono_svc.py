# Importar las bibliotecas necesarias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import joblib

# Importar módulos de scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC # Importamos Support Vector Classifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# Establecer semillas para asegurar que los resultados sean los mismos cada vez que se ejecute el script
random.seed(42)
np.random.seed(42)

# Cargar el dataset desde un archivo CSV
df = pd.read_csv('./assets/abandono_escolar.csv')

print("Primeras filas del dataset:")
print(df.head()) # Muestra las primeras 5 filas del DataFrame

print("\nValores nulos por columna:")
print(df.isnull().sum()) # Comprueba si hay valores nulos en alguna columna

# Crear un gráfico de barras para ver cuántos estudiantes abandonaron (1) y cuántos no (0)
sns.countplot(x='abandono', data=df)
plt.title('Distribución de Estudiantes que Abandonan o No')
plt.xlabel('Abandono (0 = No, 1 = Sí)')
plt.ylabel('Cantidad')
plt.tight_layout() # Ajusta automáticamente los parámetros de la figura
plt.show() # Muestra el gráfico

# Definir las variables predictoras (X) y la variable objetivo (y)
X = df.drop(columns=['abandono']) # X son todas las columnas excepto 'abandono'
y = df['abandono'] # y es la columna 'abandono'

# Escalado de variables numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Ajusta el escalador y transforma los datos

# División de los datos en conjuntos de entrenamiento y prueba
# 70% de los datos para entrenamiento y 30% para prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42 # random_state asegura la reproducibilidad de la división
)

# Entrenamiento y Optimización del Modelo SVC con GridSearchCV
print("\nIniciando búsqueda de hiperparámetros para SVC con GridSearchCV...")

# Definir la cuadrícula de parámetros a probar para el modelo SVC
param_grid_svc = {
    'C': [0.1, 1, 10, 100],  # Parámetro de regularización: penaliza los errores de clasificación
    'gamma': ['scale', 'auto', 0.1, 1], # Coeficiente del kernel RBF: define la influencia de los puntos de entrenamiento
    'kernel': ['rbf']        # Tipo de kernel: 'rbf' para manejar relaciones no lineales
}

# Configurar GridSearchCV
# estimator: el modelo a optimizar (SVC con random_state y probability=True para la curva ROC)
grid_search_svc = GridSearchCV(estimator=SVC(random_state=42, probability=True),
                               param_grid=param_grid_svc,
                               cv=5, # número de pliegues para la validación cruzada
                               scoring='f1', # métrica para evaluar la mejor combinación de parámetros (F1-score es un buen balance)
                               n_jobs=-1, # para usar todos los núcleos de la CPU disponibles
                               verbose=2) # nivel de detalle del progreso

# Entrenar GridSearchCV para encontrar la mejor combinación de hiperparámetros
grid_search_svc.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros encontrados y la mejor puntuación de validación cruzada
print(f"\nMejores hiperparámetros para SVC: {grid_search_svc.best_params_}")
print(f"Mejor puntuación F1 (de validación cruzada para SVC): {grid_search_svc.best_score_:.4f}")

# Asignar el mejor modelo SVC encontrado a la variable 'modelo'
# Esto asegura que todas las evaluaciones y visualizaciones posteriores utilicen el modelo optimizado
modelo = grid_search_svc.best_estimator_

# Evaluación del Modelo SVC
# Realizar predicciones sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular y mostrar la precisión general del modelo
print(f"\nPrecisión del modelo SVC (optimizado): {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Generar y mostrar el reporte de clasificación (precision, recall, f1-score por clase)
print("\nReporte de clasificación del modelo SVC (optimizado):")
print(classification_report(y_test, y_pred))

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Crear un mapa de calor para visualizar la matriz de confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Abandona', 'Abandona'], # Etiquetas para las predicciones
            yticklabels=['No Abandona', 'Abandona']) # Etiquetas para los valores reales
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión (Modelo SVC Optimizado)')
plt.tight_layout()
plt.show()


# Curva ROC y Área Bajo la Curva (AUC)
# Calcular las probabilidades de predicción para la clase positiva (abandono=1)
# Se requiere 'probability=True' en la inicialización de SVC para esto.
try:
    y_prob = modelo.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob) # Calcula el área bajo la curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob) # Calcula la Tasa de Falsos Positivos y Verdaderos Positivos

    # Graficar la curva ROC
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--') # Línea de referencia para un clasificador aleatorio
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC (Modelo SVC Optimizado)')
    plt.legend()
    plt.tight_layout()
    plt.show()
except AttributeError:
    print("\nNo se pudo calcular la Curva ROC: 'probability=True' no se activó en el modelo SVC.")
    print("Por favor, asegúrate de que GridSearchCV se configuró con 'estimator=SVC(random_state=42, probability=True)'")

# Guardar el modelo optimizado para poder reutilizarlo sin volver a entrenar
joblib.dump(modelo, './soluciones/ia/lab2/modelo_abandono_svc.pkl')
print("\nModelo SVC optimizado guardado como 'modelo_abandono_svc.pkl'")