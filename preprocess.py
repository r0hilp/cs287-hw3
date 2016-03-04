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
    word_to_idx = {}
    with open(word_dict, 'r') as f:
        for line in f:
            id_num, word, _ = tuple(line.split())
            word_to_idx[word] = int(id_num)
    return word_to_idx

def get_ngram_vocab(file_list, word_to_idx, ngram_size, dataset):
    ngram_to_idx = {}
    ngram_idx = 1

    for filename in file_list:
        with codecs.open(filename, "r", encoding="latin-1") as f:
            for line in f:
                line = line.rstrip()
                words = line.split()
                # Add start and end of sentence characters
                words = ['<s>' for _ in range(ngram_size - 1)] + words
                words = words + ['</s>']
                for i in range(ngram_size-1, len(words)):
                    ngram_prefix = tuple([word_to_idx[word] for word in words[i-ngram_size+1: i]])
                    if ngram_prefix not in ngram_to_idx:
                        ngram_to_idx[ngram_prefix] = ngram_idx
                        ngram_idx += 1

    return ngram_to_idx

def convert_text(data_name, word_to_idx, ngram_to_idx, ngram_size, dataset):
    ngram_features = []
    ngram_lbls = []

    with open(data_name, "r") as f:
        for line in f:
            line = line.rstrip()
            words = line.split()
            # Add start and end of sentence characters
            words = ['<s>' for _ in range(ngram_size - 1)] + words
            words = words + ['</s>']
            for i in range(ngram_size-1, len(words)):
                ngram_suffix = words[i]
                ngram_prefix = tuple([word_to_idx[word] for word in words[i-ngram_size+1: i]])
                ngram_lbls.append(word_to_idx[ngram_suffix])
                ngram_features.append(ngram_to_idx[ngram_prefix])

    return np.array(ngram_features, dtype=np.int32), np.array(ngram_lbls, dtype=np.int32)

def convert_blanks(data_name, word_to_idx, ngram_to_idx, ngram_size, dataset):
    ngram_features = []
    ngram_lbls = []
    ngram_queries = []

    with codecs.open(data_name, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip()
            words = line.split()
            if words[0] == 'Q':
                words = words[1:]
                ngram_queries.append([word_to_idx[word] for word in words])
            elif words[0] == 'C':
                words = words[1:]
                words = ['<s>' for _ in range(ngram_size - 1)] + words
                ngram = words[-ngram_size:]
                ngram_prefix = tuple([word_to_idx[word] for word in ngram[:-1]])
                ngram_features.append(ngram_to_idx[ngram_prefix])
                ngram_suffix = ngram[-1:][0]
                if ngram_suffix in word_to_idx:
                    ngram_lbls.append(word_to_idx[word])
                else:
                    ngram_lbls.append(0)

    return np.array(ngram_features, dtype=np.int32), np.array(ngram_lbls, dtype=np.int32), np.array(ngram_queries, dtype=np.int32)

FILE_PATHS = {"PTB": ("data/train.txt",
                      "data/valid.txt",
                      "data/valid_blanks.txt",
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
    train, valid, valid_blanks, test_blanks, word_dict = FILE_PATHS[dataset]

    ngram_size = args.ngram_size

    # Retrive word to id mapping
    print 'Get word ids...'
    word_to_idx = get_word_ids(word_dict)

    # Get ngram to id mapping
    print 'Get ngram ids...'
    ngram_to_idx = get_ngram_vocab([train, valid], word_to_idx, ngram_size, dataset)

    # Convert data
    print 'Processing data...'
    train_input, train_output = convert_text(train, word_to_idx, ngram_to_idx, ngram_size, dataset)

    if valid:
        valid_input, valid_output = convert_text(valid, word_to_idx, ngram_to_idx, ngram_size, dataset)

    if valid_blanks:
        valid_blanks_input, valid_blanks_output, valid_blanks_queries = convert_blanks(valid_blanks, word_to_idx, ngram_to_idx, ngram_size, dataset)

    if test_blanks:
        test_blanks_input, _, test_blanks_queries = convert_blanks(test_blanks, word_to_idx, ngram_to_idx, ngram_size, dataset)

    V = len(ngram_to_idx)
    print('# of ngrams:', V)

    C = len(word_to_idx)
    print('# of unigrams:', C)

    print 'Saving...'
    filename = args.dataset + '_' + str(ngram_size) + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if valid_blanks:
            f['valid_blanks_input'] = valid_blanks_input
            f['valid_blanks_output'] = valid_blanks_output
            f['valid_blanks_queries'] = valid_blanks_queries
        if test_blanks:
            f['test_blanks_input'] = test_blanks_input
            f['test_blanks_queries'] = test_blanks_queries
        f['nfeatures'] = np.array([V], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
