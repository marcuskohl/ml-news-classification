import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Downloading NLTK data
nltk.download('punkt')
nltk.download('stopwords')

#Loading dataset
df = pd.read_csv('../data/raw/WELFake_Dataset.csv')

#Cleaning (removing NaNs and duplicates)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

#Text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

#Applying preprocessing function to text column
df['text'] = df['text'].apply(preprocess_text)

#Exporting preprocessed data to CSV file
df.to_csv('../data/processed/welfake_pre_nltk', index=False)

#Feature extraction with TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X = tfidf_vectorizer.fit_transform(df['text'])
y = df['label']

#Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initializing logistic regression model
lr_model = LogisticRegression(random_state=42)

#Training model
lr_model.fit(X_train, y_train)

#Predicting on the test set
y_pred = lr_model.predict(X_test)

#Evaluating model
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

#Displaying performance metrics
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_mat}')
print(f'Classification Report:\n{class_report}')