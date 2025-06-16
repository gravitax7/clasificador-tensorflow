import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import json

def load_saved_model(model_dir="models"):
    try:
        # Cargar modelo
        model_path = os.path.join(model_dir, "best_model.h5")
        model = load_model(model_path)
        # Cargar nombres de clases
        metadata_path = os.path.join(model_dir, "class_mapping.json")
        with open(metadata_path, 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        print(f"Error al cargarl el modelo : {str(e)}")
        raise
##        
def preprocess_image(img_path, target_size=(128,128)):
    try:
        # Abrir imagen y convertir a RGB
        img = Image.open(img_path).convert('RGB')
        
        # Redimensionar manteniendo relación de aspecto
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Crear imagen del tamaño exacto con fondo blanco
        new_img = Image.new('RGB', target_size, (255, 255, 255))
        new_img.paste(img, ((target_size[0]-img.size[0])//2, 
                          (target_size[1]-img.size[1])//2))
        
        # Convertir a array y normalizar
        img_array = np.array(new_img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error al preprocesar {img_path}: {str(e)}")
        raise
##
def predict_image(model, class_names, img_path, input_shape):
    """Realiza la predicción y muestra los resultados"""
    try:
        input_shape = model.input_shape[1:3]
        # Preprocesar imagen
        img = preprocess_image(img_path, input_shape)
        
        # Hacer predicción
        predictions = model.predict(img)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) 
        predicted_class = class_names[predicted_class_idx]  
        
        # Mostrar imagen
        plt.figure()
        plt.imshow(image.load_img(img_path))
        plt.axis('off')
        plt.title(f"Predicción: {predicted_class} ({confidence:.2%})")
        plt.show()
        
        # Mostrar resultados detallados
        print("\nResultado de la clasificación:")
        print(f"Objeto predicho: {predicted_class}")
        print(f"Confianza: {confidence:.2%}")
        
        # Mostrar top 3 predicciones
        top_3 = np.argsort(predictions[0])[-3:][::-1]
        print("\nTop 3 predicciones:")
        for i in top_3:
            print(f"- {class_names[i]}: {predictions[0][i]:.2%}")
    except Exception as e:
        print("El modelo espera:", model.input_shape)
        img = Image.open("C:\\Users\\ROG M16\\Documents\\tensorF_desde0\\data\\raw\\TAZA\\taza_3.jpg")
        print("Tu imagen tiene:", img.size)

if __name__ == "__main__":
    # Cargar modelo entrenado
    model, class_names = load_saved_model()
    
    # Obtener forma de entrada del modelo
    input_shape = model.layers[0].input_shape[1:3]  # Obtener (alto, ancho)
    
    # Solicitar ruta de la imagen al usuario
    img_path = input("Introduce la ruta completa de la imagen a predecir: ")
    
    # Verificar que la imagen existe
    if not os.path.exists(img_path):
        print(f"Error: No se encontró el archivo {img_path}")
    else:
        try:
            predict_image(model, class_names, img_path, input_shape)
        except Exception as e:
            print(f"Error al procesar la imagen: {e}")