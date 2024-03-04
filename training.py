import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import pickle

# Cada ejemplo está en una línea y está etiquetado al final GOODUMLTRUE(BUENAS PRACTICAS) GOODUMLFALSE(MALAS PRACTICAS)
data_path = 'plantuml_dataset.txt'

# Leer los datos
with open(data_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Preprocesar los datos
# Separar los diagramas UML del etiquetado
diagrams = [re.sub(r'GOODUML( TRUE| FALSE)$', '', line).strip() for line in lines]  # Remover el etiquetado del final
labels = [1 if 'GOODUMLTRUE' in line else 0 for line in lines]  # Obtener las etiquetas

# Dividir los datos en conjuntos de entrenamiento y prueba
diagrams_train, diagrams_test, labels_train, labels_test = train_test_split(diagrams, labels, test_size=0.2, random_state=42)

# Crear un modelo de clasificación usando un pipeline
# Este pipeline convierte el texto en una matriz de conteo de tokens y luego entrena un clasificador Naive Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB(alpha=0.001))  # Agregamos un valor de alfa más bajo para suavizado

# Entrenar el modelo
model.fit(diagrams_train, labels_train)

# Evaluar el modelo
predicted_labels = model.predict(diagrams_test)
print(classification_report(labels_test, predicted_labels))

# Guardar el modelo entrenado para uso futuro
model_path = 'plantuml_model.pkl'
with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)
