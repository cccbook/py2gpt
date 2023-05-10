# gensim

* https://radimrehurek.com/gensim/models/word2vec.html (重要)
* https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
* https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

## pretrained.py

```
PS D:\pmedia\陳鍾誠\課程\人工智慧\10-lang\nn\01-word2vec\01-gensim> python pretrained.py

model.most_similar('twitter')=[('facebook', 0.9480050802230835), ('tweet', 0.9403423070907593), ('fb', 0.9342359900474548), ('instagram', 0.9104824066162109), ('chat', 0.8964964747428894), ('hashtag', 0.8885937333106995), ('tweets', 0.8878158330917358), ('tl', 0.8778460621833801), ('link', 0.8778210878372192), ('internet', 0.8753897547721863)]
model.most_similar(positive=['woman', 'king'], negative=['man'])=[('meets', 0.8841924071311951), ('prince', 0.832163393497467), ('queen', 0.8257461190223694), ('’s', 0.8174097537994385), ('crow', 0.813499391078949), ('hunter', 0.8131037950515747), ('father', 0.8115834593772888), ('soldier', 0.81113600730896), ('mercy', 0.8082393407821655), ('hero', 0.8082264065742493)]
```
