from tensorflow.keras import layers, Input, Model
from tensorflow.keras import losses, optimizers


import dill as pickle

class BiLSTM(object) : 

    def __init__(self, input_seq_len, embedding_size, vocab_size=10000, text_vectorizer_path='models/text_vectorizer.pkl' , mode=None) : 

        self.input_seq_len = input_seq_len
        self.mode = mode
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_dim = 256

        self.input = Input(shape = (self.input_seq_len, ))
        self.output = self._build_model()

    def _build_model(self) : 

        embedding = layers.Embedding(self.vocab_size, self.embedding_size)(self.input)
        bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences = True))(embedding)
        bilstm_out = layers.Bidirectional(layers.LSTM(128))(bilstm)

        return bilstm_out


    