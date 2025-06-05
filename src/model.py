from tensorflow.keras import layers, models

class CNNModel:
    def __init__(self, input_shape, num_classes):
        """
        Args:
            input_shape: Forma de la imagen (alto, ancho, canales).
            num_classes: Número de clases de salida.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        """Construye un modelo CNN básico."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model