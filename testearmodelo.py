import re
import pickle

# Cargar el modelo entrenado desde el archivo guardado
model_path = 'plantuml_model.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Función para predecir si un diagrama UML sigue buenas prácticas o no
def clasificar_diagrama(diagrama):
    # Usar el modelo para predecir si el nuevo diagrama UML sigue buenas prácticas o no
    prediction = model.predict([diagrama])

    # Interpretar la predicción
    if prediction[0] == 1:
        return "El diagrama UML sigue buenas prácticas."
    else:
        return "El diagrama UML no sigue buenas prácticas."

# Ejemplo de un nuevo diagrama UML para clasificar
# nuevo_diagrama = "@startuml Company2 User Product2 Company2 User @enduml"
nuevo_diagrama = "@startuml class Person { - name: String - age: int + setName(name: String): void + setAge(age: int): void + getName(): String + getAge(): int } class Address { - street: String - city: String - zipCode: String + setStreet(street: String): void + setCity(city: String): void + setZipCode(zipCode: String): void + getStreet(): String + getCity(): String + getZipCode(): String } Person *-- Address @enduml"
# Clasificar el nuevo diagrama UML
resultado = clasificar_diagrama(nuevo_diagrama)
print(resultado)
