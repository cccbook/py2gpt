import torch
import os

# idx : 詞彙代號 word: 詞彙
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
    # tokens: 文章長度
    def get_data(self, path, batch_size=20):
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words: 
                    self.dictionary.add_word(word)  
        print('tokens=', tokens)
        # Tokenize the file content
        ids = torch.LongTensor(tokens)
        print('len(ids)=', len(ids))
        token = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        print('ids.size(0)=', ids.size(0))
        print('batch_size=', batch_size)
        num_batches = ids.size(0) // batch_size
        print('num_batches=', num_batches)
        ids = ids[:num_batches*batch_size]
        print('len(ids)=', len(ids))
        return ids.view(batch_size, -1)