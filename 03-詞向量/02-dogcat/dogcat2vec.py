# 來源 -- https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
# Python program to generate word vectors using Word2Vec 
  
import gensim
from gensim.models import Word2Vec 

sample = open("dogcat.txt", "r") 
s = sample.read() 
lines = s.split('\n')
data = []
for line in lines: 
    if len(line) > 0:
        data.append(line.split(' ')) 

print('data=', data)
# model = gensim.models.Word2Vec(data, min_count = 1, vector_size = 2, window = 1, sg=1, negative=3) 
# model = gensim.models.Word2Vec(data, min_count = 1)
model = Word2Vec(min_count=1) # vector_size = 2, window = 1
model.build_vocab(data)
model.train(data, total_examples=model.corpus_count, epochs=100)

words = ['a', 'dog', 'cat', 'chase', 'eat', 'the']
for word in words:
	vector = model.wv[word]  # get numpy vector of a word
	# print(f'{word} vector: {vector}')
	print(f'{word}')
	sims = model.wv.most_similar(word, topn=3)  # get other similar words
	print(f'    similars:{sims}')

