from gensim import corpora, models, similarities, downloader

# Stream a training corpus directly from S3.
corpus = corpora.MmCorpus("s3://path/to/corpus")

# Train Latent Semantic Indexing with 200D vectors.
lsi = models.LsiModel(corpus, num_topics=200)

# Convert another corpus to the LSI space and index it.
index = similarities.MatrixSimilarity(lsi[another_corpus])

# Compute similarity of a query vs indexed documents.
sims = index[query]