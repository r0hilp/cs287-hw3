#!/usr/bin/env python

"""Language modeling preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re

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
        with open(filename, "r") as f:
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

def convert_text(data_name, word_to_idx, ngram_to_idx, ngram_size, dataset, context_size=5):
    ngram_features = []
    ngram_context = []
    ngram_lbls = []

    padding = max(ngram_size - 1, context_size)
    with open(data_name, "r") as f:
        for line in f:
            line = line.rstrip()
            words = line.split()
            # Add start and end of sentence characters
            words = ['<s>' for _ in range(padding)] + words
            words = words + ['</s>']
            for i in range(padding, len(words)):
                ngram_suffix = words[i]
                ngram_prefix = [word_to_idx[word] for word in words[i-ngram_size+1: i]]
                context = [word_to_idx[word] for word in words[i-context_size:i]]
                ngram_lbls.append(word_to_idx[ngram_suffix])
                ngram_features.append(ngram_to_idx[tuple(ngram_prefix)])
                ngram_context.append(context)

    return np.array(ngram_features, dtype=np.int32), np.array(ngram_context, dtype=np.int32), np.array(ngram_lbls, dtype=np.int32)

def convert_blanks(data_name, word_to_idx, ngram_to_idx, ngram_size, dataset, context_size=5):
    ngram_features = []
    ngram_lbls = []
    ngram_queries = []
    ngram_context = []
    ngram_index = []

    padding = max(ngram_size - 1, context_size)
    with open(data_name, "r") as f:
        for line in f:
            line = line.rstrip()
            words = line.split()
            if words[0] == 'Q':
                words = words[1:]
                ngram_queries.append([word_to_idx[word] for word in words])
            elif words[0] == 'C':
                words = words[1:]
                words = ['<s>' for _ in range(padding)] + words
                ngram = words[-ngram_size:]
                ngram_prefix = [word_to_idx[word] for word in ngram[:-1]]
                context = [word_to_idx[word] for word in words[-(context_size+1):-1]]
                ngram_features.append(ngram_to_idx[tuple(ngram_prefix)])
                ngram_context.append(context)

                # True word.
                word = ngram[-1]
                if word in word_to_idx:
                    ngram_lbls.append(word_to_idx[word])
                    # Get correct word index in query
                    ngram_index.append(ngram_queries[-1].index(word_to_idx[word]) + 1)
                else:
                    ngram_lbls.append(0)
                    ngram_index.append(0)

    return np.array(ngram_features, dtype=np.int32), np.array(ngram_lbls, dtype=np.int32), np.array(ngram_queries, dtype=np.int32), np.array(ngram_context, dtype=np.int32), np.array(ngram_index, dtype=np.int32)

FILE_PATHS = {"PTB": ("data/train.txt",
                      "data/valid.txt",
                      "data/valid_blanks.txt",
                      "data/test_blanks.txt",
                      "data/words.dict"),
              "PTB_small": ("data/train.1000.txt",
                            "data/valid.1000.txt",
                            None, None,
                            "data/words.1000.dict"),
              "PTB_tiny": ("data/tiny.train", "data/tiny.valid",
                           "data/tiny.valid.blanks", None, "data/words.dict")}
args = {}

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    parser.add_argument('--ngram_size', default=2, help="Ngram size", type=int)
    parser.add_argument('--context_size', default=5, help="context size", type=int)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, valid_blanks, test_blanks, word_dict = FILE_PATHS[dataset]

    ngram_size = args.ngram_size
    context_size = args.context_size

    # Retrive word to id mapping
    print 'Get word ids...'
    word_to_idx = get_word_ids(word_dict)

    # Get ngram to id mapping
    print 'Get ngram ids...'
    ngram_to_idx = get_ngram_vocab([train, valid], word_to_idx, ngram_size, dataset)

    # Convert data
    print 'Processing data...'
    train_input, train_context, train_output = convert_text(train, word_to_idx, ngram_to_idx, ngram_size, dataset, context_size)

    if valid:
        valid_input, valid_context, valid_output = convert_text(valid, word_to_idx, ngram_to_idx, ngram_size, dataset, context_size)

    if valid_blanks:
        valid_blanks_input, valid_blanks_output, valid_blanks_queries, valid_blanks_context, valid_blanks_index = convert_blanks(valid_blanks, word_to_idx, ngram_to_idx, ngram_size, dataset, context_size)

    if test_blanks:
        test_blanks_input, _, test_blanks_queries, test_blanks_context, _ = convert_blanks(test_blanks, word_to_idx, ngram_to_idx, ngram_size, dataset, context_size)

    num_ngrams = len(ngram_to_idx)
    print('# of ngrams:', num_ngrams)

    vocab_size = len(word_to_idx)
    print('# of unigrams:', vocab_size)

    print 'Saving...'
    filename = args.dataset + '_' + str(ngram_size) + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_context'] = train_context
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_context'] = valid_context
            f['valid_output'] = valid_output
        if valid_blanks:
            f['valid_blanks_input'] = valid_blanks_input
            f['valid_blanks_output'] = valid_blanks_output
            f['valid_blanks_queries'] = valid_blanks_queries
            f['valid_blanks_context'] = valid_blanks_context
            f['valid_blanks_index'] = valid_blanks_index
        if test_blanks:
            f['test_blanks_input'] = test_blanks_input
            f['test_blanks_queries'] = test_blanks_queries
            f['test_blanks_context'] = test_blanks_context
            f['test_blanks_index'] = test_blanks_index
        f['ngrams_size'] = np.array([num_ngrams], dtype=np.int32)
        f['vocab_size'] = np.array([vocab_size], dtype=np.int32)
        f['context_size'] = np.array([context_size], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
