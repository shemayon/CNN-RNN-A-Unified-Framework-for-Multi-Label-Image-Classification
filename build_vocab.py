import os
import pickle
import argparse
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

    def __len__(self):
        return len(self.word2idx)

    def to_index(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx['<unk>']

    def to_word(self, index):
        if index < len(self.idx2word):
            return self.idx2word[index]
        else:
            return '<unk>'
# class Vocabulary:
#     def __init__(self):
#         self.word2index = {}
#         self.index2word = {}
#         self.vocab_size = 0

#     def add_word(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.vocab_size
#             self.index2word[self.vocab_size] = word
#             self.vocab_size += 1

#     def load(self, vocab_path):
#         with open(vocab_path, 'rb') as f:
#             vocab = pickle.load(f)
#             self.word2index = vocab['word2index']
#             self.index2word = vocab['index2word']
#             self.vocab_size = vocab['vocab_size']

#     def save(self, vocab_path):
#         with open(vocab_path, 'wb') as f:
#             pickle.dump({'word2index': self.word2index, 'index2word': self.index2word, 'vocab_size': self.vocab_size}, f)

# Example random word frequency variable for demonstration
word_frequency = {
    'apple': 1500,
    'banana': 1200,
    'orange': 1000,
    'mango': 800,
    'kiwi': 600,
    'pineapple': 400,
    'pear': 200
}

def build_vocab(json, threshold, word_frequency):
    words = [word for word, cnt in word_frequency.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for word in words:
        vocab.add_word(word)
    return vocab

def main(args):
    if not os.path.exists('data'):  # Check if the 'data' directory exists
        os.makedirs('data')  # If not, create the 'data' directory

    vocab = build_vocab(json=args.caption_path, threshold=args.threshold, word_frequency=word_frequency)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
