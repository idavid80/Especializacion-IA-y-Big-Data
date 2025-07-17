from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time

# Configuración de Selenium con ChromeDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# cargar página web
url = "https://books.toscrape.com/"
driver.get(url)

# Extraer los elementos con información de los libros
books = driver.find_elements(By.CLASS_NAME, "product_pod")
data = []

# Recorrer la lista de libros y extraer título y precio

for book in books:
    title = book.find_element(By.TAG_NAME, 'h3').text
    price = book.find_element(By.CLASS_NAME, 'price_color').text
    data.append([title, price])

# Convertir la lista de datos a un DataFrame de Pandas
df = pd.DataFrame(data, columns=["Título", "Precio"])

# Cerrar el navegador
driver.quit()

# Guardar los datos en un archivo CSV
df.to_csv("./soluciones/bigdata/lab5/precios_libros.csv", index=False)

print("Scraping completado. Datos guardados en precios_libros.csv")