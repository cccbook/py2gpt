

```
$ python corpus2vec.py english
model= Word2Vec<vocab=12, vector_size=100, alpha=0.025>
words= dict_keys(['the', 'little', 'pig', 'every', 'white', 'cat', 'chase', 'a', 'bite', 'black', 'dog', 'love'])

the
    similars:[('every', 0.9970623254776001), ('a', 0.993272602558136), ('black', 0.8270807862281799)]
little
    similars:[('white', 0.9918485879898071), ('black', 0.9839619398117065), ('cat', 0.8558139204978943)]
pig
    similars:[('dog', 0.9984816312789917), ('cat', 0.9965192079544067), ('little', 0.8464518189430237)]
every
    similars:[('the', 0.9970623254776001), ('a', 0.9963570237159729), ('black', 0.8339701294898987)]
white
    similars:[('black', 0.9962167143821716), ('little', 0.9918486475944519), ('love', 0.8452091217041016)]
cat
    similars:[('pig', 0.9965193867683411), ('dog', 0.9950042366981506), ('little', 0.8558138608932495)]
chase
    similars:[('bite', 0.9977421760559082), ('love', 0.9959043860435486), ('black', 0.8492988348007202)]
a
    similars:[('every', 0.9963570833206177), ('the', 0.993272602558136), ('black', 0.8490776419639587)]
bite
    similars:[('chase', 0.9977419972419739), ('love', 0.9973382353782654), ('black', 0.843748152256012)]
black
    similars:[('white', 0.9962170720100403), ('little', 0.9839619398117065), ('love', 0.8618208765983582)]
dog
    similars:[('pig', 0.9984820485115051), ('cat', 0.995004415512085), ('little', 0.8470156192779541)]
love
    similars:[('bite', 0.9973384737968445), ('chase', 0.9959043860435486), ('black', 0.8618208169937134)]
```