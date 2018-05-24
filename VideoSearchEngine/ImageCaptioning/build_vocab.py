import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
from tqdm import tqdm

class Vocabular(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    '''
    create a vocabulary out of the json object and remove the words
    that occur under a certain threshold
    '''
    coco = COCO(json) # build the COCO object instance
    counter = Counter()
    ids = coco.anns.keys()
    for i, coco_id in enumerate(tqdm(ids)):
        caption = str(coco.anns[coco_id]['caption'])

        # lets tokenize the caption
        caption_tokens = nltk.tokenize.word_tokenize(caption.lower())

        counter.update(caption_tokens)
    
    # eliminate words that occur under a threshold
    words = [word for word, count in counter.items() if count >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary
    for i, word in enumerate(words):
        vocab.add_word(word)
    
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


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