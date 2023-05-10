# 來源 -- https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
# Python program to generate word vectors using Word2Vec 
  
import gensim
from gensim.models import Word2Vec 
from sklearn.decomposition import PCA
from matplotlib import pyplot

sample = open("dogcat.txt", "r") 
s = sample.read() 
lines = s.split('\n')
data = []
for line in lines: 
    if len(line) > 0:
        data.append(line.split(' ')) 

print('data=', data)
model = Word2Vec(min_count=1) # vector_size = 2, window = 1
model.build_vocab(data)
model.train(data, total_examples=model.corpus_count, epochs=100)
# model = gensim.models.Word2Vec(data, min_count = 1, vector_size = 2, window = 1, sg=1, negative=3) 
# model = gensim.models.Word2Vec(data, min_count = 1, vector_size = 2, window = 1, sg=1, negative=2, epochs=100) 

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