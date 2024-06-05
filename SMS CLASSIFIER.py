import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv(r'C:\Users\User\Desktop\spam or not.CSV', encoding='latin-1')
# Check the structure of the dataset
print(df.head())

# Map 'spam' to 1 and 'ham' to 0
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(df['v2'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, predictions))
