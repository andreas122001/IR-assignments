import numpy as np
from nltk.stem.porter import PorterStemmer
import string


class CustomIndexer:
    def __init__(self, corpus) -> None:
        self.original_corpus = corpus
        self.preprocessed_corpus = [self.preprocessing(doc) for doc in corpus]
        self.index = self.gen_idx_block(self.preprocessed_corpus)

    def search(self, query):
        query = self.preprocessing(query)
        all_docs = {}
        for q in self.preprocessing(query).split(" "):
            if q in self.index:
                all_docs[q] = self.index[q]        
        all_docs = list([list(val.keys()) for val in all_docs.values()])
        docs = []
        for doc in all_docs[0]:
            if all([doc in temp for temp in all_docs]):
                docs.append(doc)
        return docs
        

    def blockify(self, corpus, block_size=3):
        corpus = [[doc.split(" ")[block_size*i:block_size*i+block_size] for i in range(len(doc.split(" "))//block_size+1)] for doc in corpus]
        corpus = [list(filter(lambda x: len(x) > 0, doc)) for doc in corpus]
        return corpus

    def gen_idx_block(self, corpus, block_size=3):
        # Initiate the index as a dict('term', dict('doc', [block_ids]))
        idx_list = dict([(key, {}) for key in set(" ".join(corpus).split(" "))])
        corpus_blocks = []
        for doc_idx, doc in enumerate(corpus, 1):
            # Generate blocks
            blocks = [doc.split(" ")[block_size*i:block_size*i+block_size] for i in range(len(doc.split(" "))//block_size+1)]
            blocks = list(filter(lambda x: len(x)>0, blocks))
            corpus_blocks.append(blocks)
            # For each distinct term in the document
            for term in set(doc.split(" ")):
                if doc_idx not in idx_list[term].keys():
                    idx_list[term][doc_idx] = []
                # Find occurrences and add block to block list:
                for block_idx, block in enumerate(blocks):
                    if term in block:
                        idx_list[term][doc_idx].append(block_idx)
        return idx_list


    # This is used for preprocessing of both the corpus and queries
    def preprocessing(self, text):
        # Initiate stemmer
        stemmer = PorterStemmer()

        # Define unwanted characters (punctuation)
        bad_chars = string.punctuation+"\n\r\t"

        # Clean, tokenize and stem text
        new_text = text = text.lower() # all lower case
        new_text = "".join(list(filter(lambda x: x not in bad_chars, new_text))) # remove unwanted chars
        new_text = new_text.split(" ") # tokenize (split into words)
        new_text = list(filter(lambda c: len(c) > 0, new_text)) # remove empty strings
        new_text = [stemmer.stem(word) for word in new_text] # perform stemming
        new_text = " ".join(new_text)
        return new_text
    
    def retrieve_raw_indexes(self, terms):
        indexes = []
        for term in terms:
            if term in self.index:
                indexes.append(self.index[term])
        return indexes
    
    def get_docs_by_union(terms):
        # Get the indexes of the NOT terms
        indexes = retrieve_raw_indexes(terms)
        if not len(indexes):
            return []
        # Accumulate docs of the NOT terms
        docs = set()
        for index in indexes:
            docs.update(index.keys())
        return list(docs)


corpus = []
for i in range(1,7):
    with open(f"./DataAssignment4/Text{i}.txt") as f:
        corpus.append(f.read())

indexer = CustomIndexer(corpus)
print(indexer.search("enjoy"))
