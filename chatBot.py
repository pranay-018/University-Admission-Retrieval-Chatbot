import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading the database
df = pd.read_csv('new_faq.csv', delimiter=",")
df.dropna(inplace=True)

# Check the column names to ensure they are correct
print(df.columns)

# Initializing the vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(np.concatenate((df['Question'], df['Answer'])))
question_vector = vectorizer.transform(df['Question'])

# Chat with the user
print("Now you can chat with the user")
while True:
    input_question = input('>> ')
    if input_question.lower() == 'quit':
        break
    else:
        input_question_vector = vectorizer.transform([input_question])
        similarities = cosine_similarity(input_question_vector, question_vector)
        closest = np.argmax(similarities, axis=1)
        print("BOT: " + df['Answer'].iloc[closest[0]])
