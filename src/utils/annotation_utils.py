import json
import os
from typing import Text
import dill as pickle

from tqdm import tqdm
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf


input_sequence_len = 30
vocabulary_size = 10000
embedding_size = 256

annotation_dir = 'data/annotation'
splits = ['train', 'val']

vectorizer_path = 'models/text_vectorizer.pkl'



def prepare_text_vectorizer(vocabulary_size, input_sequence_len, save_path) : 

    vectorizer = TextVectorization(max_tokens=vocabulary_size, output_sequence_length=input_sequence_len)
    corpus = []
    for split in splits : 

        with open(os.path.join(annotation_dir, '{}.json'.format(split))) as f : 
            annotations = json.load(f)

        num_annotations = len(annotations)

        for i in tqdm(range(num_annotations), desc='{} Split'.format(split)) : 

            corpus.append(annotations[i]['question'])
            for answer in annotations[i]['answers'] : 
                corpus.append(answer['answer'] )

    vectorizer.adapt(corpus, batch_size=1024)

    print("Finished training the vectorizer. Here is a test run for the sentence Life, The Universe and Everything")
    print(vectorizer("Life, The Universe and Everything"))
    
    with open (save_path, 'wb') as p : 
        pickle.dump({'config':vectorizer.get_config(),
                    'weights':vectorizer.get_weights()}, p)

    print('Finished saving to : ' , save_path)

def read_text_vectorizer(save_path) : 

    with open(save_path, 'rb') as p :

        vectorizer_metadata = pickle.load(p)
        vectorizer = TextVectorization.from_config(vectorizer_metadata['config'])

        # vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        # added above line because of actual bug in keras.
        # ignore this line of code. needs to be there to make stuff work.
        # not removing it because I am not sure what version actually fixes this bug. 
          
        vectorizer.set_weights(vectorizer_metadata['weights'])

    print("Finished retrieving the vectorizer. Here is a test run for the sentence : Life, The Universe and Everything")
    print(vectorizer("Life, The Universe and Everything"))

    return vectorizer



if __name__ == '__main__' : 


    prepare_text_vectorizer(vocabulary_size, input_sequence_len, save_path=vectorizer_path)
    read_text_vectorizer(save_path=vectorizer_path)