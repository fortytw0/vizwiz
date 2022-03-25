import json
import glob
import string
import numpy as np


output_chars = list(string.ascii_lowercase + string.digits + string.punctuation + ' ')
output_sequence_len = 50
output_vector_size = len(output_chars) + 1



def char_encode(label) : 

    label = label.lower() # explicitly making the label lowercase

    if len(label) > output_sequence_len : # truncating label
        label = label[:output_sequence_len]

    encoded = np.zeros((output_sequence_len, output_vector_size))

    for i, char in enumerate(label) : 

        char_vector = np.zeros((output_vector_size, ))
        if char in output_chars : 
            char_index = output_chars.index(char) # known character
            char_vector[char_index] = 1.0
        else : 
            char_vector[-1] = 1.0 # unknown character

        encoded[i] = char_vector

    return encoded

if __name__ == '__main__' : 

    print(output_chars)
    print('Encoding : "abc.-!@# " ')
    print(char_encode("abc.-!@# "))
    print(char_encode("abc.-!@# ").shape)