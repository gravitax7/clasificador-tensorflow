import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import json

def load_saved_model(model_dir="models"):
    """Carga el modelo y los nombres de clases"""
    model_path = os.path.join(model_dir, "best_model.h5")
    metadata_path = os.path.join(model_dir, "class_mapping.json")
    
    model = load_model(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print("Salidas del modelo:", model.output_shape)    
    class_names = metadata["classes"]
    return model, class_names

def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocesa la imagen para el modelo"""
    try:
        # Cargar imagen y convertir a RGB
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
        print(f"Error procesando imagen: {str(e)}")
        raise

def predict_and_display(model, class_names, img_path):
    """Realiza la predicción y muestra los resultados"""
    try:
        # Preprocesar imagen
        processed_img = preprocess_image(img_path)
        
        # Realizar predicción
        predictions = model.predict(processed_img)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        
        # Mostrar comparación de imágenes
        plt.figure(figsize=(12, 6))
        
        # Imagen original
        plt.subplot(1, 2, 1)
        original_img = Image.open(img_path)
        plt.imshow(original_img)
        plt.title(f"Original\nTamaño: {original_img.size}")
        plt.axis('off')
        
        # Imagen preprocesada
        plt.subplot(1, 2, 2)
        plt.imshow(processed_img[0])  # Mostrar primera (y única) imagen del batch
        plt.title(f"Preprocesada (128×128)\nPredicción: {predicted_class} ({confidence:.2%})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Mostrar resultados detallados en consola
        print("\nRESULTADOS DE CLASIFICACIÓN")
        print("="*40)
        print(f"Clase predicha: {predicted_class} (Confianza: {confidence:.2%})")
        print(f"Índice de clase: {predicted_class_idx}")
        
        # Mostrar top 5 predicciones
        print("\nTOP 5 PREDICCIONES:")
        sorted_indices = np.argsort(predictions[0])[::-1]  # Orden descendente
        for i, idx in enumerate(sorted_indices[:5]):
            print(f"{i+1}. {class_names[idx]}: {predictions[0][idx]:.2%}")
            
    except Exception as e:
        print(f"\nERROR durante la predicción: {str(e)}")

if __name__ == "__main__":
    try:
        # Cargar modelo
        print("Cargando modelo...")
        model, class_names = load_saved_model()
        print(f"Modelo cargado. Clases disponibles: {class_names}")
        
        # Obtener ruta de imagen
        img_path = input("\nIntroduce la ruta completa de la imagen: ").strip('"')
        
        # Verificar que existe
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No se encontró el archivo: {img_path}")
            
        # Realizar predicción
        print("\nProcesando imagen...")
        predict_and_display(model, class_names, img_path)
        
    except Exception as e:
        print(f"\nError general: {str(e)}")