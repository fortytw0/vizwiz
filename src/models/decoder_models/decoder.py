from tensorflow.keras import layers, Input, Model
from tensorflow.keras import losses, optimizers

class Decoder(object) : 

    def __init__(self, encoder_output, image_model_output_size, question_model_output_size, output_sequence_len=50, output_vector_size=69) :

        self.image_model_output_size = image_model_output_size
        self.question_model_output_size = question_model_output_size
        self.output_sequence_len = output_sequence_len
        self.output_vector_size = output_vector_size

        self.input = layers.Dense(64, activation='relu')(encoder_output)
        self.output = self._build_model()


    def _build_model(self) : 

        
        dense = layers.Dense(512, activation='relu')(self.input)
        dense = layers.Dropout(0.2)(dense)
        dense = layers.BatchNormalization()(dense)
        
        dense = layers.Dense(128, activation='relu')(dense)
        dense = layers.Dropout(0.2)(dense)
        dense = layers.BatchNormalization()(dense)

        dense = layers.Dense(64, activation='relu')(dense)

        repeat = layers.RepeatVector(self.output_sequence_len)(dense)
        bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences = True))(repeat)
        bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences = True))(bilstm)
        output = layers.LSTM(self.output_vector_size, activation='softmax', return_sequences = True)(bilstm)

        return output
        



