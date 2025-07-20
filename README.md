# Especializacón IA y Big-Data

Curso de Especialización de Inteligencia Artificial y Big Data

## Descripción

Este repositorio contiene algunos de los ejercicios de laboratorios que he realizado durante el curso oficial de Inteligencia Artificial y Big Data. Los laboratorios que no aparecen en este repositorio, es porque son ejercicios realizados en plataformas o herramientas como AWS, Make, Power BI.

## Dependencias

```bash
pip install pandas
pip install selenium
pip install scikit-learn
pip install matplotlib
pip install nltk
pip install seaborn
```

## Laboratorios Big Data

### 1. Implementación de un flujo ETL en Python

Este laboratorio tiene como objetivo comprender y aplicar el proceso ETL (Extracción, Transformación y Carga) utilizando Python, basado en un caso práctico de un ecommerce en crecimiento. Aprenderemos a manipular datos de un archivo CSV, transformarlos para análisis y cargarlos en una base de datos SQLite.

#### 5. Predicción de Precios en Marketplaces con Cloud Computing, Web Scraping, Machine Learning, Automatización y Visualización.

Desarrollar un sistema automatizado para la extracción y análisis de precios en marketplaces. Se utilizará web scraping para recolectar información de productos desde una tienda en línea, se almacenarán los datos en Google Sheets, y se aplicará un modelo de regresión lineal para predecir la evolución de los precios. Todo el proceso será gestionado tanto de forma local, como a través de computación en la nube y mediante la automatización de Make en la nube. Los resultados se visualizarán a través de alguna herramienta como Power BI, Tableau, Airtable o similar.

[API: Tienda de libros](https://books.toscrape.com/)

Estructura de la página.

![Estructura de la página](./images/scrapping.png)

## Laboratorios Inteligencia Artificial

### 1. Análisis de Sentimientos para Reseñas de Productos clasificar reseñas de productos en positivas o negativas

Aprender a construir un modelo de inteligencia artificial sencillo utilizando Python. El objetivo es clasificar reseñas de productos en positivas o negativas, entendiendo el proceso paso a paso y el porqué de cada acción.

Los resultados obtenidos son muy interesantes y reflejan los desafíos comunes con conjuntos de datos pequeños en el análisis de sentimientos.

Tamaño del Dataset y sus Implicaciones
Entrenamiento: 35 muestras

Prueba: 15 muestras

Este es el principal factor que limita el rendimiento de tus modelos. Con solo 35 reseñas para entrenar, los modelos tienen muy poca información para aprender los patrones complejos del lenguaje y el sentimiento. Es como intentar aprender un idioma completo leyendo solo 35 frases.

De los tres modelos con los que se ha realizado el laboratorio, el MLP muestra un balance mucho mejor entre las clases. Aunque su exactitud general es similar a MNB, no tiene una polarización tan fuerte hacia una clase. Es capaz de identificar mejor los negativos (5 correctos vs 3 de MNB y 2 de LR) pero introduce algunos falsos negativos.

<div style="text-align:center"><img src="./soluciones/ia/lab1/lab1_matriz.png" /></div>

<div style="text-align:center"><img src="./soluciones/ia/lab1/lab1_barras.png" /></div>

### 2. Predicción del Abandono Escolar

La propuesta es utilizar Machine Learning para desarrollar un modelo que prediga el riesgo de abandono escolar basado en datos históricos de los alumnos, permitiendo a las escuelas tomar medidas preventivas de manera anticipada.

Los resultados de este laboratorio nos indica que con el modelo de clasificación Random Forest obtenemos una precisión general del 64.33%. Es ligeramente mejor para predecir abandono (Clase 1) en términos de precisión (0.70) y un poco mejor para identificar no abandono (Clase 0) en términos de recall (0.69).

Aunque la precisión para la clase de abandono es buena, el recall de 0.60 significa que el modelo aún pasa por alto un porcentaje significativo de estudiantes que realmente abandonan. Esto podría ser crítico si el objetivo es intervenir proactivamente.

![metodo codo](./soluciones/ia/lab2/matriz.png) ![score](./soluciones/ia/lab2/caracteristicas.png)
![metodo codo](./soluciones/ia/lab2/curva%20ROC.png)

Para continuar con el laboratorio, se han empleado otros modelos como:

- **Grid Search**
Mejor puntuación F1 (de validación cruzada): 0.7025
Precisión del modelo Grid Search: 68.00%

Reporte de clasificación Grid Search:

| Clase           | Precisión | Recall | F1-score | Soporte |
| --------------- | --------- | ------ | -------- | ------- |
| No abandono (0) | 0.62      | 0.74   | 0.68     | 136     |
| Abandono (1)    | 0.75      | 0.63   | 0.68     | 164     |
| **Accuracy**    |           |        | **0.68** | **300** |

### 4. Segmentación de clientes: k-means

El objetivo de la actividad es aplicar el algoritmo de clustering K-means para segmentar a los clientes según sus hábitos de compra y proponer estrategias de marketing dirigidas a cada segmento identificado.

**Conclusiones Generales:**

- Segmento 1 (mayores) es el más numeroso y de buen valor: ideal para retención y consolidación.
- Segmento 2 (jóvenes) tiene el mayor ticket promedio: una excelente oportunidad para crecimiento futuro.
- Segmento 0 representa un grupo menos comprometido, ideal para estrategias de reactivación o recuperación.

| Segmento | Edad promedio | Compra promedio | Compras anuales | Nº de clientes |
| -------- | ------------- | --------------- | --------------- | -------------- |
| **0**    | 40.33 años    | \$153.16        | 1.55            | 12 clientes    |
| **1**    | 61.72 años    | \$272.42        | 2.30            | 25 clientes    |
| **2**    | 28.92 años    | \$299.64        | 2.23            | 13 clientes    |

![metodo codo](./soluciones/ia/lab4/metodo_codo.png) ![score](./soluciones/ia/lab4/score.png)

![metodo codo](./soluciones/ia/lab4/szegmentacion.png) ![score](./soluciones/ia/lab4/szegmentacion3d.png)
