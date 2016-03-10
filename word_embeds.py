import h5py
import numpy as np
import operator
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_word_ids(word_dict):
    # Construct word to PTB id mapping
    word_to_idx = {}
    idx_to_word = []
    with open(word_dict, 'r') as f:
        for line in f:
            id_num, word, _ = tuple(line.split())
            word_to_idx[word] = int(id_num)
            idx_to_word.append(word)
    return word_to_idx, idx_to_word

def get_closest(w, word_to_idx, idx_to_word, embeds):
    e = embeds[word_to_idx[w] - 1]
    dots = []
    for i,r in enumerate(embeds):
        if i < 50 or idx_to_word[i] == w:
            continue
        dots.append((idx_to_word[i], np.dot(e, r)))

    dots.sort(key=operator.itemgetter(1))
    dots.reverse()
    print dots[:10]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('word', help="word", type=str)
    parser.add_argument('--pca_words', type=str)
    args = parser.parse_args()

    with h5py.File('word_embeds.hdf5', 'r') as f:
        word_to_idx, idx_to_word = get_word_ids('data/words.dict')
        embeds = np.array(f['embeds'])

        get_closest(args.word, word_to_idx, idx_to_word, embeds)

        pca = PCA(n_components=2)
        pca.fit(embeds)
        embeds = pca.transform(embeds)
        subset = [word_to_idx[w]-1 for w in args.pca_words.split(',')]

        plt.figure()
        plt.scatter(embeds[subset,0], embeds[subset,1])
        for i in subset:
            plt.annotate(idx_to_word[i], (embeds[i,0], embeds[i,1]))
        plt.show()

        
if __name__ == '__main__':
    main()

