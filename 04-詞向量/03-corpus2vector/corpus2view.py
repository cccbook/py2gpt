# 來源 -- https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
# Python program to generate word vectors using Word2Vec 
  
import gensim
from gensim.models import Word2Vec
import sys
from sklearn.decomposition import PCA
from matplotlib import pyplot

corpus = sys.argv[1]
sample = open(f"corpus/{corpus}.txt", mode="r", encoding="utf8") 
s = sample.read() 
lines = s.split('\n')
data = []
vocab = {}
for line in lines: 
	if len(line) <= 0: continue
	vocs = line.split(' ')
	data.append(vocs)
	for voc in vocs:
		if voc in vocab:
			vocab[voc] = vocab[voc]+1
		else:
			vocab[voc] = 1


# model = gensim.models.Word2Vec(data, min_count = 1, vector_size = 2, window = 1, sg=1, negative=3) 
# model = gensim.models.Word2Vec(data, min_count = 1)
model = Word2Vec(min_count=1) # vector_size = 2, window = 1
model.build_vocab(data)
model.train(data, total_examples=model.corpus_count, epochs=100)
print('model=', model)

words = vocab.keys()
print('words=', words)
for word in words:
	vector = model.wv[word]  # get numpy vector of a word
	# print(f'{word} vector: {vector}')
	print(f'{word}')
	sims = model.wv.most_similar(word, topn=3)  # get other similar words
	print(f'    similars:{sims}')


# fit a 2d PCA model to the vectors
X = model.wv[model.wv.key_to_index]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.key_to_index)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

'''
words = ['a', 'dog', 'cat', 'chase', 'eat', 'the']
for word in words:
	vector = model.wv[word]  # get numpy vector of a word
	print(f'{word} vector: {vector}')
	sims = model.wv.most_similar(word, topn=3)  # get other similar words
	print(f'    similars:{sims}')
'''