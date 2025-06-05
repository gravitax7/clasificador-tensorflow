import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from .data_loading import ImageDatasetLoader
from .model import CNNModel

class Trainer:
    def __init__(self, data_dir, model_dir="models"):
        """
        Args:
            data_dir: Ruta a los datos de entrenamiento.
            model_dir: Carpeta para guardar modelos.
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)  # Crea carpeta si no existe

    def train(self, epochs=10, batch_size=32):
        """Entrena el modelo y guarda checkpoints."""
        # Carga datos
        loader = ImageDatasetLoader(self.data_dir)
        X, y = loader.load_data()
        
        # Construye modelo
        model = CNNModel(input_shape=X[0].shape,num_classes=len(np.unique(y)))
        model = model.build_model()

        # Callback para guardar el mejor modelo
        checkpoint_path = os.path.join(self.model_dir, "best_model.h5")
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )

        # Entrenamiento
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[checkpoint]
        )
        
        return history
    
if __name__ == "__main__":
    trainer = Trainer(data_dir="data/raw")
    trainer.train()