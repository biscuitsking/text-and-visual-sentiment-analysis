import nltk
import pickle
import argparse
from collections import Counter
import json




class Vocabulary(object):
    #
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self,word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self,word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json_path, threshold):
    #建立词典
    counter = Counter()
    with open(json_path, 'r') as f:
        sentence_dic = json.load(f)
        f.close()

    for ii,key in enumerate(sentence_dic):
        sentence = sentence_dic[key]
        tokens =nltk.tokenize.word_tokenize(sentence.lower())
        counter.update(tokens)

        if (ii+1) % 1000 == 0:
            print('[{}/{}] Tokenized.'.format(ii,len(sentence_dic)))

    words = [word for word,cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')

    for i,word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(json_path=args.caption_path,threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path,'wb') as f:
        pickle.dump(vocab,f)
        f.close()
    print('Total vocabulary size : {}'.format(len(vocab)))
    print("save vocab to '{}'".format(vocab_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path',type=str,
                        default='/home/theo/Xli/sentenses.json',
                        help= 'path for sentences file')
    parser.add_argument('--vocab_path',type=str,
                        default='./vocab.pkl',
                        help='path for save vocab')
    parser.add_argument('--threshold',type=int,default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
