import os
import time

from src.models.model1 import CBD
from src.utils.train_utils import TrainGenerator

from tensorflow.keras import losses, optimizers, callbacks


train_data = TrainGenerator('train')
val_data = TrainGenerator('val')
epochs = 10

model_dir = 'models/'
log_dir = 'logs/'

cbd = CBD('models/', 'logs/')

cbd.model.summary()

print('Compiling model : ')
cbd.model.compile(loss=losses.BinaryCrossentropy(), optimizer=optimizers.Adam())
print('Succesfully compiled model')

model_ckpt = callbacks.ModelCheckpoint(os.path.join(model_dir, '{epoch:02d}-{val_loss:.2f}.hdf5'))
csv_logging = callbacks.CSVLogger(os.path.join(log_dir, 'train_{}.log'.format(time.time())))

[print(i.shape, i.dtype) for i in cbd.model.inputs]
[print(o.shape, o.dtype) for o in cbd.model.outputs]


generator = train_data.generator()

X, Y = next(generator)

print(X[0].shape)
print(X[1].shape)
print(Y.shape)

cbd.model.predict(X, batch_size=32)


# cbd.model.fit(x=train_data.generator())

# history = cbd.model.fit(x=train_data.generator(), 
#             batch_size=train_data.batch_size, 
#             steps_per_epoch=train_data.steps_per_epoch, 
#             callbacks= [model_ckpt, csv_logging],
#             epochs=epochs,
#             validation_data=val_data.generator(),
#             validation_batch_size=val_data.batch_size,
#             validation_steps=val_data.steps_per_epoch)

# print(type(history))