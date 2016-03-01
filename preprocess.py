#!/usr/bin/env python

"""Language modeling preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

# Your preprocessing, features construction, and word2vec code.

def get_word_ids(word_dict):
    # Construct word to PTB id mapping
    word_to_id = {}
    with open(word_dict, 'r') as f:
        for line in f:
            id_num, word, _ = tuple(line.split())
            word_to_id[word] = int(id_num)
    return word_to_id

def convert_data(data_name, word_to_idx, ngram_size, dataset):
    # Construct index feature sets for each file.
    features = []
    ngram_features = []
    ngram_lbls = []

    with codecs.open(data_name, "r", encoding="latin-1") as f:
        for line in f:
            # Start of sentence padding
            features.extend([1] * (ngram_size - 1))
            words = line.split(' ')
            for word in words:
                features.append(word_to_idx[word])
            # End of sentence padding
            features.append(2)

    # Construct ngram windows
    for i in range(len(features)):
        # Skip padding
        if features[i] == 1:
            continue
        else:
            ngram_features.append(features[i-ngram_size+1:i])
            ngram_lbls.append(features[i])

    return np.array(ngram_features, dtype=np.int32), np.array(ngram_lbls, dtype=np.int32)

def get_vocab(file_list, dataset=''):
    # Construct index feature dictionary.
    word_to_idx = {}
    # Start at 3 (1 is padding, 2 is end of sentence)
    idx = 3
    for filename in file_list:
        if filename:
            with codecs.open(filename, "r", encoding="latin-1") as f:
                for line in f:
                    words = line.split(' ')
                    for word in words:
                        if word not in word_to_idx:
                            word_to_idx[word] = idx
                            idx += 1
    return word_to_idx

FILE_PATHS = {"PTB": ("data/train.txt",
                      "data/valid.txt",
                      "data/test_blanks.txt",
                      "data/words.dict")}
args = {}

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    parser.add_argument('ngram_size', default=2, help="Ngram size", type=int)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test, word_dict = FILE_PATHS[dataset]

    ngram_size = args.ngram_size

    # Retrive word to id mapping
    print 'Get word ids...'
    word_to_id = get_word_ids(word_dict)

    # Get index features
    print 'Getting vocab...'
    word_to_idx = get_vocab([train, valid, test], dataset)

    # Convert data
    print 'Processing data...'
    train_input, train_output = convert_data(train, word_to_idx, ngram_size, dataset)

    if valid:
        valid_input, valid_output = convert_data(valid, word_to_idx, ngram_size, dataset)

    if test:
        test_input, test_output = convert_data(test, word_to_idx, ngram_size, dataset)

    # -1 for start of sentence padding
    C = len(word_to_idx) - 1
    print('Vocab size:', C)

    print 'Saving...'
    filename = args.dataset + '_' + str(ngram_size) + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
        f['nclasses'] = np.array([C], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
