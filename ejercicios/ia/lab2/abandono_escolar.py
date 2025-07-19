import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# Reproducibilidad total
random.seed(42)
np.random.seed(42)

# Cargar datos
df = pd.read_csv('./assets/abandono_escolar.csv')
print("Primeras filas del dataset:")
print(df.head())
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Distribución de la variable objetivo
sns.countplot(x='abandono', data=df)
plt.title('Distribución de Estudiantes que Abandonan o No')
plt.xlabel('Abandono (0 = No, 1 = Sí)')
plt.ylabel('Cantidad')
plt.tight_layout()
plt.show()

# Variables predictoras y objetivo
X = df.drop(columns=['abandono'])
y = df['abandono']

# Escalado de variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Entrenamiento modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluación
y_pred = modelo.predict(X_test)

print(f"\nPrecisión del modelo: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

#  Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Abandona', 'Abandona'],
            yticklabels=['No Abandona', 'Abandona'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.tight_layout()
plt.show()

# Importancia de variables
importances = modelo.feature_importances_
features = X.columns

sns.barplot(x=importances, y=features)
plt.title('Importancia de Características')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.tight_layout()
plt.show()


# Curva ROC y AUC
y_prob = modelo.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend()
plt.tight_layout()
plt.show()

# Guardar modelo
joblib.dump(modelo, './soluciones/ia/lab2/modelo_abandono.pkl')
print("\nModelo guardado como 'modelo_abandono.pkl'")
