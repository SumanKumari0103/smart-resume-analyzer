import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# dataset load
data = pd.read_csv("dataset.csv")

# input output
x = data["skills"]
y = data["role"]

# text convert
cv = CountVectorizer()
x_vector = cv.fit_transform(x)

# model train
model = MultinomialNB()
model.fit(x_vector, y)

# save model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(cv, open("model/vectorizer.pkl", "wb"))

print("Model Trained Successfully")