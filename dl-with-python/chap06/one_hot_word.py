import numpy as np
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework']
def numpy_process():
    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    print(token_index)
    max_length = 10
    results = np.zeros(shape=(len(samples), 
                              max_length,
                              max(token_index.values()) + 1))
    print(results.shape)

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            idx = token_index[word]
            results[i, j, idx] = 1
    print(results)

def keras_process():
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(samples)
    sequences = tokenizer.texts_to_sequences(samples)
    print(sequences)

    return

if __name__ == '__main__':
    print(samples)
    numpy_process()
    keras_process()

