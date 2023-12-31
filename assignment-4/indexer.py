import numpy as np
from nltk.stem.porter import PorterStemmer
import string
import re
import codecs
from tqdm import tqdm

class Parser:
    def parse_query(self, query):
        # Tokenize the query
        query = query.replace(" -", " NOT ")
        tokens = re.findall(r'(?:AND|OR|NOT|\(|\)|[a-zA-Z0-9]+)', query)
        return self.__parse_expression(tokens)

    def __parse_expression(self, tokens):
        current_expression = []

        while tokens:
            token = tokens.pop(0)
            if token == "(":
                # Start a new nested expression
                nested_expression = self.__parse_expression(tokens)
                current_expression.append(nested_expression)
            elif token == ")":
                # End the current expression
                break
            else:
                current_expression.append(token)
        return current_expression

def read_stopwords():
    common_words = []
    try:
        with codecs.open("common-english-words.txt", "r", "utf-8") as f:
            common_words = f.read().split(",")
    except:
        print("Could not read stopwords, continuing without.")
    return common_words

class CustomIndexer:
    def __init__(self, corpus) -> None:
        self.stopwords = read_stopwords()
        self.corpus = corpus
        self.preprocessed_corpus = [self.preprocessing(doc) for doc in tqdm(corpus)]
        self.index = self.gen_idx(self.preprocessed_corpus)
        self.parser = Parser()
        

    def blockify(self, corpus, block_size=3):
        corpus = [[doc.split(" ")[block_size*i:block_size*i+block_size] for i in range(len(doc.split(" "))//block_size+1)] for doc in corpus]
        corpus = [list(filter(lambda x: len(x) > 0, doc)) for doc in corpus]
        return corpus
    
    def gen_idx(self, corpus):
        # Initiate the index as a dict('term', dict('doc', num_occ))
        idx_list = dict([(key, {}) for key in set(" ".join(corpus).split(" "))])
        for doc_idx, doc in enumerate(tqdm(corpus), 0):
            # Increment number of occurrences for each occurrence
            for term in doc.split(" "):
                if doc_idx not in idx_list[term].keys():
                    idx_list[term][doc_idx] = 0
                idx_list[term][doc_idx] += 1
        return idx_list

    def gen_idx_block(self, corpus, block_size=3):
        # Initiate the index as a dict('term', dict('doc', [block_ids]))
        idx_list = dict([(key, {}) for key in set(" ".join(corpus).split(" "))])
        corpus_blocks = []
        for doc_idx, doc in enumerate(tqdm(corpus), 0):
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
        new_text = list(filter(lambda c: c not in self.stopwords, new_text)) # remove empty strings
        new_text = [stemmer.stem(word) for word in new_text] # perform stemming
        new_text = " ".join(new_text)
        return new_text
        
    def __retrieve_docs(self, term):
        term = self.preprocessing(term)
        if term in self.index:
            return set(self.index[term].keys())
        return set()
    
    def filter_recursive(self, query):
        current_context = set()
        operator = None
        doc_set = set()

        for term in query:
            if isinstance(term, list):
                # Recursively process sub-expression
                sub_result = self.filter_recursive(term)
                if operator:
                    if operator == 'AND':
                        current_context.intersection_update(sub_result)
                    elif operator == 'NOT':
                        current_context.difference_update(sub_result)
                    else: # default OR
                        current_context.update(sub_result)
                else:
                    current_context.update(sub_result)
            elif term in {'AND', 'OR', 'NOT', '-'}:
                # Set the current operator
                operator = term
            else:
                # Process term
                term_docs = self.__retrieve_docs(term)  # Replace with your actual retrieval function
                if operator:
                    if operator == 'AND':
                        current_context.intersection_update(term_docs)
                    elif operator == 'OR':
                        current_context.update(term_docs)
                    elif operator == 'NOT':
                        current_context.difference_update(term_docs)
                    operator = None
                else:
                    current_context.update(term_docs)
        
        # After processing all terms and operators in the query, update the document set
        doc_set.update(current_context)
        return doc_set
    
    def rank_docs(self, docs, query):
        # Generate vocabulary of all retrieved documents
        vocab = set(" ".join([self.preprocessed_corpus[doc] for doc in docs]).split(" "))

        # Remove terms that are not in vocabulary from query
        query = self.preprocessing(query).split(" ")
        query = list(filter(lambda x: x in vocab, query))
        # Remove operators so they don't influence rank
        query = list(filter(lambda x: x not in ['and', 'or', 'not'], query))

        # Calculate TF-IDF
        tfs = np.array([[(self.index[term][doc] if doc in self.index[term] else 0) for term in query] for doc in docs])
        dfs = np.array([len(self.index[term]) for term in query])
        idfs = np.log2(len(self.corpus) / dfs)
        tf_idf = np.sum(tfs*idfs, axis=1)

        # Gather results and sort by score
        result = []
        for doc, score in zip(docs, tf_idf):
            result.append({
                'id': doc,
                'score': score,
                'content': self.corpus[doc]
            })
        result = sorted(result, key=lambda x: x['score'])[::-1]

        return result

    def search(self, query):
        parsed_query = self.parser.parse_query(query)
        filtered_results = list(self.filter_recursive(parsed_query))
        if not filtered_results:
            return None
        ranked_results = self.rank_docs(filtered_results, query)
        return ranked_results

# E.g.: fox AND brown AND lazy OR (claim -morning)
if __name__=="__main__":
    corpus = []
    for i in range(1,7):
        with open(f"./DataAssignment4/Text{i}.txt") as f:
            corpus.append(f.read())

    indexer = CustomIndexer(corpus)
    query = "0"
    while query:
        query = input("Query: ")

        results = indexer.search(query)
        if results:
            for res in results:
                print(f"ID: {res['id']}, Score: {res['score']:.4f}, Content: {res['content'][:100]}")
        else:
            print("No results")
        print()
