import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report



~ Question 1
df = pd.read_csv("recipes.csv")

~identifying missing data and cleaning

~ showing summary statistics
print(df[["rating_avg", "rating_val"]].describe())

~ displaying 10 highest rated recipes
print("\n")
print(df.nlargest(10, 'rating_avg')["title"])
print("\n")
print(df.nlargest(10, 'rating_val')["title"])

~ Question 2
~ visualizing data
df.plot(x="rating_avg", y=["rating_val"], kind="scatter",)
~ df.plot(x="rating_avg", y=["rating_val"])
plt.show()

~ commenting of relationship or rating_val and rating_avg
print("\n")
print("The higher the rating_val the lower the rating_avg")
print("The threshold i suggest is 3 since 25% of rating_val is 3.\nTherefore any rating_val below or equal to 3 can be considered not significant")

print("\n")
~ Question 3
features=['title','rating_avg','rating_val','total_time','category','cuisine', 'ingredients']


combined_features = []


for i in df.index:
        fresh = df.loc[i, features].values.flatten().tolist()
            new = " ".join(str(x) for x in fresh)
                combined_features.append(new)
                
                
                ~ adding combine features to dataframe
                df['combine_features'] = combined_features
                
                
                ~ Defining vec_space_method
                def vec_space_method(recipe_query):
                        document = df["combine_features"]
                            ~ print(document)
                            
                            
                                ~using the countVectorizer class
                                    ~ Create a Vectorizer Object
                                        vectorizer = CountVectorizer()
                                        
                                            vectorizer.fit(document)
                                            
                                                ~ Printing the identified Unique words along with their indices
                                                    ~ print("Vocabulary: ", vectorizer.vocabulary_)
                                                    
                                                        ~ Encode the Document
                                                            vector = vectorizer.transform(document)
                                                                ~ print(vector)
                                                                
                                                                
                                                                
                                                                    vectorizer = TfidfVectorizer()
                                                                        X = vectorizer.fit_transform(document)
                                                                            ~ print(vectorizer.get_feature_names())
                                                                                ~ print(X.shape)
                                                                                
                                                                                    vector = X
                                                                                        df1 = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names())
                                                                                            ~ print(df1)
                                                                                            
                                                                                            
                                                                                                nltk.download('punkt')
                                                                                                    nltk.download('stopwords')
                                                                                                    
                                                                                                    
                                                                                                        ~ print(stopwords.words('english'))
                                                                                                            stop_words = set(stopwords.words('english'))
                                                                                                            
                                                                                                            
                                                                                                                def get_tokenized_list(doc_text):
                                                                                                                            tokens = nltk.word_tokenize(doc_text)
                                                                                                                                    return tokens
                                                                                                                                
                                                                                                                                    ~ This function will performing stemming on tokenized words
                                                                                                                                        def word_stemmer(token_list):
                                                                                                                                                  ps = nltk.stem.PorterStemmer()
                                                                                                                                                        stemmed = []
                                                                                                                                                              for words in token_list:
                                                                                                                                                                          stemmed.append(ps.stem(words))
                                                                                                                                                                                return stemmed
                                                                                                                                                                            
                                                                                                                                                                                ~ Function to remove stopwords from tokenized word list
                                                                                                                                                                                    def remove_stopwords(doc_text):
                                                                                                                                                                                              cleaned_text = []
                                                                                                                                                                                                    for words in doc_text:
                                                                                                                                                                                                                if words not in stop_words:
                                                                                                                                                                                                                              cleaned_text.append(words)
                                                                                                                                                                                                                                    return cleaned_text
                                                                                                                                                                                                                                
                                                                                                                                                                                                                                
                                                                                                                                                                                                                                    ~Check for single document
                                                                                                                                                                                                                                        tokens = get_tokenized_list(document[1])
                                                                                                                                                                                                                                            ~ print("WORD TOKENS:")
                                                                                                                                                                                                                                                ~ print(tokens)
                                                                                                                                                                                                                                                    doc_text = remove_stopwords(tokens)
                                                                                                                                                                                                                                                        ~ print("\nAFTER REMOVING STOPWORDS:")
                                                                                                                                                                                                                                                            ~ print(doc_text)
                                                                                                                                                                                                                                                                ~ print("\nAFTER PERFORMING THE WORD STEMMING::")
                                                                                                                                                                                                                                                                    doc_text = word_stemmer(doc_text)
                                                                                                                                                                                                                                                                        ~ print(doc_text)
                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                            doc_ = ' '.join(doc_text)
                                                                                                                                                                                                                                                                                ~ print(doc_)
                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                    cleaned_document = []
                                                                                                                                                                                                                                                                                        for doc in document:
                                                                                                                                                                                                                                                                                                  tokens = get_tokenized_list(doc)
                                                                                                                                                                                                                                                                                                        doc_text = remove_stopwords(tokens)
                                                                                                                                                                                                                                                                                                              doc_text  = word_stemmer(doc_text)
                                                                                                                                                                                                                                                                                                                    doc_text = ' '.join(doc_text)
                                                                                                                                                                                                                                                                                                                          cleaned_document.append(doc_text)
                                                                                                                                                                                                                                                                                                                              ~ print(cleaned_document)
                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                  vectorizerX = TfidfVectorizer()
                                                                                                                                                                                                                                                                                                                                      vectorizerX.fit(cleaned_document)
                                                                                                                                                                                                                                                                                                                                          doc_vector = vectorizerX.transform(cleaned_document)
                                                                                                                                                                                                                                                                                                                                              ~ print(vectorizerX.get_feature_names())
                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                  ~ print(doc_vector.shape)
                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                      df1 = pd.DataFrame(doc_vector.toarray(), columns=vectorizerX.get_feature_names())
                                                                                                                                                                                                                                                                                                                                                          ~ print(df1)
                                                                                                                                                                                                                                                                                                                                                          