from prediccion2 import predict_from_base64, load_saved_model
import json
import os

if __name__ == "__main__":
    try:
        ##print("Directorio actual:", os.getcwd())
        model, class_names = load_saved_model()
        ##print("Modelo cargado.")

        with open(r"C:\Users\ROG M16\Documents\tensorF_desde0\src\imagen.txt", "r") as f:
            base64_str = f.read()
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",")[1]

        resultado = predict_from_base64(model, class_names, base64_str)
        print(json.dumps(resultado, ensure_ascii=False, indent=4))

    except Exception as e:
        print(f"\nError general: {str(e)}")