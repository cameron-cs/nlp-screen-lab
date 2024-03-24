import argparse
import os

import numpy as np
from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
from nltk import word_tokenize


import paths
from glove_util import transfer
from train import load_data, remove_alpha_and_symbols_description, stopwords_filtering


def encode_text(text, model):
    tokens = word_tokenize(text)
    vectors = [model[word] for word in tokens if word in model]
    if vectors:
        return sum(vectors) / len(vectors)
    else:
        return np.zeros(model.vector_size)


def index_data(sqlite_file):
    try:
        raw_tvmaze = load_data(sqlite_file, 'tvmaze')
        print(f"Successfully read data from the input sqlite file: ", sqlite_file)
        glove_model = None
        data = preprocess_data(raw_tvmaze)

        glove_embedding = f"{paths.glove_path}/{paths.glove_embedding_path}"
        word2vec_format = f"{paths.glove_path}/{paths.glove_word2vec_path}"
        glove_model_path = f"{paths.location_of_model}/{paths.glove_model}"
        if not os.path.exists(glove_model_path):
            print(f"Transferring the raw GloVE embeddings to Word2Vec format: from '{glove_embedding}', to '{word2vec_format}'")
            transfer(glove_embedding, word2vec_format)
            glove_model = KeyedVectors.load_word2vec_format(word2vec_format, binary=False)
        else:
            glove_model = KeyedVectors.load(glove_model_path)

        data['encoded_description'] = data['preprocessed_description'].apply(lambda x: encode_text(x, glove_model))

        data['encoded_description'] = data['encoded_description'].apply(lambda x: np.array(x, dtype=np.float32))

        data = data.dropna(subset=['encoded_description'])

        glove_model_path = f"{paths.location_of_model}/{paths.glove_model}"

        print(f"Saving the GloVE model: ", glove_model_path)
        glove_model.save(glove_model_path)

        index_path = f"{paths.location_of_index}/{paths.indexed_tv_shows}"
        print(f"Saving the index: ", index_path)
        data[["tvmaze_id", "showname", "encoded_description"]].to_parquet(index_path, index=False)

        print(f"Finished indexing...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def preprocess_data(data):
    data = data.drop_duplicates(subset=['tvmaze_id', 'description'])

    data = data[data['tvmaze_id'].notna()]

    data = data[data['description'].notna()]

    noisy_descriptions = data['description'].str.len() < 10

    data = data[~noisy_descriptions]

    data['description'] = data['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())

    data['filtered_description'] = data['description'].apply(
        lambda description: remove_alpha_and_symbols_description(description))

    data['preprocessed_description'] = data['filtered_description'].apply(lambda p: stopwords_filtering(p))

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index data from an SQLite database")
    parser.add_argument("--raw-data", required=True, help="Path to the SQLite database file")

    args = parser.parse_args()

    index_data(args.raw_data)
