from tensorflow.keras import layers, Input, Model
from tensorflow.keras import losses, optimizers


class CNN(object) : 

    def __init__(self, image_shape, mode=None) : 
        self.image_shape = image_shape
        self.mode = mode
        self.output_dim = 512

        self.input = Input(shape=self.image_shape)
        self.output = self._build_model()



    def _build_model(self) : 

        cnn = layers.Conv2D(128, (3,3), activation='relu')(self.input)
        cnn = layers.Conv2D(128, (3,3), activation='relu')(cnn)
        cnn = layers.Conv2D(64, (3,3), activation='relu')(cnn)
        cnn = layers.MaxPool2D()(cnn)
        cnn = layers.Conv2D(32, (3,3), activation='relu')(cnn)
        cnn = layers.MaxPool2D()(cnn)
        cnn = layers.Conv2D(32, (3,3), activation='relu')(cnn)
        cnn = layers.MaxPool2D()(cnn)


        flat = layers.Flatten()(cnn)

        cnn_output = layers.Dense(self.output_dim, activation='relu')(flat)

        return cnn_output



