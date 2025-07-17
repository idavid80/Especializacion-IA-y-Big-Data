import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt # For plotting confusion matrix (optional)
import seaborn as sns # For better looking plots (optional)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

archivo = "./assets/reviews.csv"
data = pd.read_csv(archivo)

# vectorizar = CountVectorizer(stop_words='english', ngram_range=(1, 2))
vectorizar = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

x = vectorizar.fit_transform(data['review'])
y = data['sentiment']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Algoritmo Multinomial Naive Bayes
model_mnb = MultinomialNB()
model_mnb.fit(x_train, y_train)

y_pred_mnb = model_mnb.predict(x_test)

accuracyy_mnb = accuracy_score(y_test, y_pred_mnb)
print(f"Precisión del modelo: {accuracyy_mnb * 100:.2f}%\n")

print("--- Informe de Clasificación ---")
print(classification_report(y_test, y_pred_mnb))


print("--- Matriz de Confusión ---")
cm = confusion_matrix(y_test, y_pred_mnb)
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model_mnb.classes_,
            yticklabels=model_mnb.classes_)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Modelo de Regresión Logística
print("--- Evaluando Modelo de Regresión Logística ---")
model_lr = LogisticRegression(max_iter=1000) # Aumentamos max_iter para asegurar convergencia
model_lr.fit(x_train, y_train)

y_pred_lr = model_lr.predict(x_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Precisión del modelo de Regresión Logística: {accuracy_lr * 100:.2f}%\n")

print("--- Informe de Clasificación (Regresión Logística) ---")
print(classification_report(y_test, y_pred_lr))

print("--- Matriz de Confusión (Regresión Logística) ---")
print(confusion_matrix(y_test, y_pred_lr))

# Modelo de Red Neuronal (MLPClassifier)
print("\n--- Evaluando Modelo de Red Neuronal (MLPClassifier) ---")
# hidden_layer_sizes: define el número de capas y neuronas por capa.
# alpha: parámetro de regularización para evitar overfitting.
# max_iter: número máximo de iteraciones.
model_mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42)
model_mlp.fit(x_train, y_train)

y_pred_mlp = model_mlp.predict(x_test)

accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"Precisión del modelo de Red Neuronal (MLP): {accuracy_mlp * 100:.2f}%\n")

print("--- Informe de Clasificación (MLP) ---")
print(classification_report(y_test, y_pred_mlp))

print("--- Matriz de Confusión (MLP) ---")
print(confusion_matrix(y_test, y_pred_mlp))

# --- Prediction on New Reviews ---
new_reviews = [
    "La calidad es fantástica, me ha encantado.",
    "No me ha gustado para nada, una experiencia terrible.",
    "Muy mal servicio, no lo recomiendo en absoluto.",
    "El producto es bueno pero el envío fue lento.",
    "Absolutamente perfecto, superó mis expectativas.",
    "Una decepción total, no volveré a comprar.",
    "Funciona como se esperaba, bastante satisfecho.",
    "Increíblemente inútil, un desperdicio de dinero."
]

new_X = vectorizar.transform(new_reviews)

# Predicciones de los modelos
predictions_mnb = model_mnb.predict(new_X)
predictions_lr = model_lr.predict(new_X)
predictions_mlp = model_mlp.predict(new_X)


# Imprimir predicciones
print("\n--- Predicciones de Nuevas Reseñas (MNB) ---")
for review, sentiment in zip(new_reviews, predictions_mnb):
    print(f"Review: \"{review}\" \nSentimiento: {sentiment}\n")

print("\n--- Predicciones de Nuevas Reseñas (Regresión Logística) ---")
for review, sentiment in zip(new_reviews, predictions_lr):
    print(f"Review: \"{review}\" \nSentimiento: {sentiment}\n")

print("\n--- Predicciones de Nuevas Reseñas (MLP) ---")
for review, sentiment in zip(new_reviews, predictions_mlp):
    print(f"Review: \"{review}\" \nSentimiento: {sentiment}\n")