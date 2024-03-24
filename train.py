import argparse
import re
import sqlite3

import joblib
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

import paths

nltk.download("stopwords")
nltk.download('wordnet')

eng_stopwords = set(stopwords.words('english'))


def lemmatization_description(description):
    lemmatizer = WordNetLemmatizer()
    filtered_text = [lemmatizer.lemmatize(x) for x in description.split()]
    return ' '.join(filtered_text)


def remove_alpha_and_symbols_description(description):
    description = re.sub("|'", "", description)
    description = re.sub("[^a-zA-Z]", " ", description)
    description = ' '.join(description.split())
    description = description.lower()

    return description


def stopwords_filtering(description):
    filtered_text = [x for x in description.split() if not x in eng_stopwords]
    return ' '.join(filtered_text)


def load_data(sqlite_file, table):
    connection = sqlite3.connect(sqlite_file)

    query = f"SELECT * FROM {table}"

    data_table = pd.read_sql_query(query, connection)

    connection.close()

    return data_table


def train_model(data):
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(data['genre'])
    y1 = multilabel_binarizer.transform(data['genre'])

    bow_vectorizer = CountVectorizer()
    X = bow_vectorizer.fit_transform(data['filtered_description'])

    lr_bow = LogisticRegression(max_iter=1000)
    clf_bow = OneVsRestClassifier(lr_bow)
    clf_bow.fit(X, y1)


def evaluate_model(model, X_test, y_test):
    # make predictions on the test set
    y_pred = model.predict(X_test)

    # calculate the accuracy of the model's predictions
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def save_model(model, path):
    # Save the trained model to paths.location_of_model
    joblib.dump(model, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model from SQLite data")
    parser.add_argument("--training-data", required=True, default='tvmaze.sqlite', help="Path to the SQLite database file")

    args = parser.parse_args()

    training_data = args.training_data

    raw_genre = load_data(training_data, 'tvmaze_genre')
    raw_tvmaze = load_data(training_data, 'tvmaze')

    print(f"Successfully read data from the input sqlite file: ", training_data)

    genre = raw_genre[raw_genre['genre'].notna()]
    tvmaze = raw_tvmaze[raw_tvmaze['description'].notna()]

    raw_merged = pd.merge(tvmaze, genre, on='tvmaze_id')

    noisy_descriptions = raw_merged['description'].str.len() < 20

    raw_merged = raw_merged[~noisy_descriptions]

    raw_merged['description'] = raw_merged['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())

    grouped_data = raw_merged.groupby(['tvmaze_id', 'description'])['genre'].agg(list).reset_index()

    filtered_data = grouped_data

    filtered_data['filtered_description'] = filtered_data['description'].apply(
        lambda description: remove_alpha_and_symbols_description(description))

    eng_stopwords = set(stopwords.words('english'))

    filtered_data['filtered_description'] = filtered_data['filtered_description'].apply(lambda p: stopwords_filtering(p))

    filtered_data['filtered_description'] = filtered_data['filtered_description'].apply(
        lambda x: lemmatization_description(x))

    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(filtered_data['genre'])
    y1 = multilabel_binarizer.transform(filtered_data['genre'])

    print(f"Training the model...")
    X_train, X_test, y_train, y_test = train_test_split(filtered_data['filtered_description'], y1, test_size=0.2,
                                                        random_state=42)

    bow_vectorizer = CountVectorizer()
    X_bow = bow_vectorizer.fit_transform(filtered_data['filtered_description'])

    lr_bow = LogisticRegression(max_iter=1000)
    model = OneVsRestClassifier(lr_bow)
    model.fit(X_bow, y1)

    y_pred = model.predict(bow_vectorizer.transform(X_test))
    accuracy = evaluate_model(model, bow_vectorizer.transform(X_test), y_test)
    print(f"Accuracy: {accuracy:.2f}")

    multilabel_binarizer_path = f'{paths.location_of_model}/{paths.binarizer_filename}'
    print(f"Saving the MultiLabelBinarizer model: ", multilabel_binarizer_path)
    save_model(multilabel_binarizer, f'{paths.location_of_model}/{paths.binarizer_filename}')

    model_path = f'{paths.location_of_model}/{paths.model_filename}'
    print(f"Saving the OneVsRestClassifier model: ", model_path)
    save_model(model, model_path)

    bow_vectorizer_path = f'{paths.location_of_model}/{paths.vectorizer_filename}'
    print(f"Saving the CountVectorizer model: ", bow_vectorizer_path)
    save_model(bow_vectorizer, bow_vectorizer_path)

    print('Models were saved: ', [multilabel_binarizer_path, model_path, bow_vectorizer_path])

    print(f"Finished training...")

