import os
import time

from tensorflow.keras import layers, Model
from tensorflow.keras import losses, optimizers, callbacks

from src.utils.image_utils import image_shape
from src.utils.annotation_utils import input_sequence_len, vocabulary_size, embedding_size, vectorizer_path
from src.utils.output_utils import output_vector_size, output_sequence_len

from src.models.image_models.CNN import CNN
from src.models.question_models.bilstm import BiLSTM
from src.models.decoder_models.decoder import Decoder


class CBD(object) : 

    def __init__(self, model_dir, log_dir) : 

        self.model_dir = model_dir
        self.log_dir = log_dir

        print('Initializing CBD model...')

        self.cnn = CNN(image_shape)
        self.bilstm = BiLSTM(input_sequence_len, embedding_size, vocabulary_size, vectorizer_path)
        
        self.encoder_outputs = layers.Concatenate()([self.cnn.output, self.bilstm.output])

        self.decoder = Decoder(self.encoder_outputs,
                            self.cnn.output_dim, 
                            self.bilstm.output_dim, 
                            output_sequence_len, 
                            output_vector_size)

        self.model = self._build_model()
        # self.callbacks = self._get_callbacks()
        # self._compile_model()


    def _build_model(self) :         
        return Model(inputs = [self.cnn.input, self.bilstm.input], outputs = self.decoder.output)

    def _compile_model(self) : 
        self.model.compile(loss=losses.BinaryCrossentropy(), optimizer=optimizers.Adam())

    def _get_callbacks(self) : 
        model_ckpt = callbacks.ModelCheckpoint(os.path.join(self.model_dir, '{epoch:02d}-{val_loss:.2f}.hdf5'))
        csv_logging = callbacks.CSVLogger(os.path.join(self.log_dir, 'train_{}.log'.format(time.time())))
        return [model_ckpt, csv_logging] 

    def train_model(self, train_data, val_data, epochs=10) : 

        self.model.fit(x=train_data.generator(), 
                    batch_size=train_data.batch_size, 
                    steps_per_epoch=train_data.steps_per_epoch, 
                    callbacks= self.callbacks,
                    epochs=epochs,
                    validation_data=val_data.generator(),
                    validation_batch_size=val_data.batch_size,
                    validation_steps=val_data.steps_per_epoch
                    )


if __name__ == '__main__' : 

    print("Whathahasiodhjalosdjklasdjkl")
    cbd = CBD('models/', 'logs/')
    cbd.model.summary()