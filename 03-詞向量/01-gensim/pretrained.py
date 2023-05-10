import gensim.downloader
# Show all available models in gensim-data
# print(list(gensim.downloader.info()['models'].keys()))
model = gensim.downloader.load('glove-twitter-25')

print(f"model.most_similar('twitter')={model.most_similar('twitter')}")
print(f"model.most_similar('dog')={model.most_similar('dog')}")
print(f"model.most_similar('mother')={model.most_similar('mother')}")
print(f"model.most_similar('king')={model.most_similar('king')}")
print(f"model.most_similar('push')={model.most_similar('push')}")
print(f"model.most_similar(positive=['woman', 'king'], negative=['man'])={model.most_similar(positive=['woman', 'king'], negative=['man'])}")
