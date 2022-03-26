from __future__ import annotations
import json
import os
import numpy as np
from tqdm import tqdm

from src.utils.image_utils import read_image, image_shape, img_dir
from src.utils.annotation_utils import annotation_dir, input_sequence_len, embedding_size, vectorizer_path, vocabulary_size, read_text_vectorizer
from src.utils.output_utils import output_sequence_len, output_vector_size, char_encode


class TrainGenerator(object) : 

    def __init__(self, split, 
                batch_size=32) -> None:
        

        self.split = split
        self.annotation_path = os.path.join(annotation_dir, '{}.json'.format(self.split))
        self.image_dir = img_dir

        print('Reading images from : {}'.format(self.annotation_path))

        with open(self.annotation_path) as f : 
            self.annotations = json.load(f)

        self.num_samples = len(self.annotations)
        self.batch_size = batch_size
        self.steps_per_epoch = int(self.num_samples/self.batch_size)

        print('Successfully read annotations.')
        print('There are {} samples in this dataset.'.format(self.num_samples))
        print('The batch size selected is : {}'.format(self.batch_size))
        print('Steps per epoch is : ', self.steps_per_epoch)
        print('WARNING: Since each answer is considered separately, \
            you can expect the actual batch to be 10x the mentioned \
            batch size. ')

        self.image_shape = image_shape

        print('Finished initializing image settings.')
        print('Image shape : {}'.format(self.image_shape))

        self.input_sequence_len = input_sequence_len
        self.embedding_size = embedding_size 
        self.vocabulary_size = vocabulary_size
        self.vectorizer_path = vectorizer_path
        self.vectorizer = read_text_vectorizer('models/text_vectorizer.pkl')

        print('Finished initializing annotation settings.')
        print('Input Sequence Length : ' , self.input_sequence_len)
        print('Embedding Size : ' , self.embedding_size)
        print('Vocabulary Size : ' , self.vocabulary_size)
        print('Vectorizer Path : ' , self.vectorizer_path)

        self.output_sequence_len = output_sequence_len
        self.output_vector_size = output_vector_size

        print('Finished initializing output settings.')
        print('Output sequence length : ' , self.output_sequence_len)
        print('Output Vector Size : ' , output_vector_size)
        
        # Iterator settings
        self.start = 0 
        self.end = self.start + self.batch_size
        self.i = 0

    def generator(self) : 

        while True : 
            
            if self.end > self.num_samples : 
                self.end = self.num_samples

            elif self.end == self.num_samples : 
                self._reset_iterators()


            annotation_batch = self.annotations[self.start:self.end]
            batch_images, batch_questions, batch_Y = self._retrieve_annotations(annotation_batch)
            if (len(batch_images) != self.batch_size*10) or (len(batch_questions) != self.batch_size*10) or (len(batch_Y) != self.batch_size*10): 
                print(len(batch_images))
                print(len(batch_questions))
                print(len(batch_Y))


            self.i += 1
            self.start = self.i * self.batch_size
            self.end = self.start + self.batch_size
            yield (np.array(batch_images), np.array(batch_questions)), np.array(batch_Y)

    def _reset_iterators(self) :
        
        self.start = 0 
        self.end = self.start + self.batch_size
        self.i = 0

    def _retrieve_annotations(self, annotation_batch) :
        batch_images = []
        batch_questions = []
        batch_Y = []
        
        for annotation in annotation_batch : 
            try :
                img = self._retrieve_image(annotation['image'])
                question = self._retrieve_question(annotation['question'])
                answers = self._retrieve_answers(annotation['answers'])

                for answer in answers : 
                    batch_images.append(img)
                    batch_questions.append(question)
                    batch_Y.append(answer)

            except Exception as e: 
                print(e)
                pass

        return batch_images, batch_questions, batch_Y


    def _retrieve_image(self, image_name) : 
        return read_image(image_name, self.split)

    def _retrieve_question(self, question) : 
        return self.vectorizer(question).numpy()

    def _retrieve_answers(self, answers) : 
        return [char_encode(answer['answer']) for answer in answers]


if __name__ == '__main__'  :

    train_data = TrainGenerator('train')
    data_gen = train_data.generator()
    print(data_gen)

    for i in range(10) : 
        batch = next(data_gen)
        
        inputs , answer = batch
        img, question = inputs

        print('Image shape : ' , img.shape)
        print('Question shape : ' , question.shape)
        print('Answer shape : ' , answer.shape)

        print('Generator Iteration : {}/{}'.format(train_data.i, train_data.steps_per_epoch))
        print('Generator start pointer : ' , train_data.start)
        print('Generator end pointer : ' , train_data.end)

    train_data = TrainGenerator('train')
    data_gen = train_data.generator()
    for i in tqdm(range(train_data.steps_per_epoch + 10)) : 
        next(data_gen)

