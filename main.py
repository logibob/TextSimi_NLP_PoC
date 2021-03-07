# based on
# https://dev.to/coderasha/compare-documents-similarity-using-python-nlp-4odp

from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import gensim

# Only once necessary?!
#nltk.download('punkt')

file_docs = []

#data = "Mars is approximately half the diameter of Earth."
#print(word_tokenize(data))

data = "Hello Mars! Mars is a cold desert world. It is half the size of Earth. "
print(sent_tokenize(data))

tokens = sent_tokenize(data)
for line in tokens:
    file_docs.append(line)

print("Number of documents:",len(file_docs))

gen_docs = [[w.lower() for w in word_tokenize(text)] 
    for text in file_docs]

dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary.token2id)

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
print(corpus)

tf_idf = gensim.models.TfidfModel(corpus)
for doc in tfidf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])