import argparse
import json
import sys

import pandas as pd
from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity

import paths
from index import encode_text
from train import lemmatization_description, stopwords_filtering, remove_alpha_and_symbols_description


def read_index(path):
    return pd.read_parquet(path)


def read_file(file_path, encoding):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        print(f"Failed to decode the file using the '{encoding}' encoding.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)


def search_tv_shows(input_file, output_json_file, encoding='UTF-8'):
    try:
        print('Loading the input file: ', input_file)
        raw_query = read_file(input_file, encoding)

        index_file_path = f"{paths.location_of_index}/{paths.indexed_tv_shows}"

        print('Loading the index file: ', index_file_path)
        indexed_tv_shows = read_index(index_file_path)

        query = lemmatization_description(stopwords_filtering(remove_alpha_and_symbols_description(raw_query.lower())))

        glove_model_path = f"{paths.location_of_model}/{paths.glove_model}"

        print('Loading the GloVE model: ', glove_model_path)
        glove_model = KeyedVectors.load(glove_model_path)

        encoded_query = encode_text(query, glove_model)

        print(f'Searching similar descriptions for the query "{raw_query}" ...')
        similarities = indexed_tv_shows['encoded_description'].apply(lambda x: cosine_similarity([encoded_query], [x])[0][0])

        indexed_tv_shows['similarity'] = similarities
        indexed_tv_shows = indexed_tv_shows.sort_values(by='similarity', ascending=False)

        top_matchings = 3
        matchings = 0

        matched_shows = []

        for index, row in indexed_tv_shows.iterrows():
            if matchings < top_matchings:
                matched_shows += {"tvmaze_id": row['tvmaze_id'], "showname": row['showname']},
                matchings += 1
            else:
                break

        print('Saving the matched shows: ', output_json_file)
        with open(output_json_file, 'w+', encoding='UTF-8') as json_file:
            json.dump(matched_shows, json_file, ensure_ascii=False)
        print(f"Finished searching...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for TV shows based on a query")
    parser.add_argument("--input-file", required=True, help="Path to the input file with the search query")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for matched shows")
    parser.add_argument("--encoding", default="UTF-8", help="Input file encoding (default: UTF-8)")

    args = parser.parse_args()

    search_tv_shows(args.input_file, args.output_json_file, args.encoding)
