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
cuisine = df['cuisine'].replace(r'^\s*$', np.nan, regex=True)

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

