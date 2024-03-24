import argparse
import json
import os
import re
import sys

import joblib
import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

import paths

nltk.download("stopwords")
nltk.download('wordnet')

eng_stopwords = set(stopwords.words('english'))


def filter_description(description):
    description = re.sub("|'", "", description)
    description = re.sub("[^a-zA-Z]", " ", description)
    description = ' '.join(description.split())
    description = description.lower()

    return description


def lemmatization_description(description):
    lemmatizer = WordNetLemmatizer()
    filtered_text = [lemmatizer.lemmatize(x) for x in description.split()]
    return ' '.join(filtered_text)


def stopwords_filtering(description):
    filtered_text = [x for x in description.split() if not x in eng_stopwords]
    return ' '.join(filtered_text)


def explain_genre_predictions_txt(original_text, filtered_text, predicted_labels, genre_probabilities,
                                  top_words_per_label, output_txt_file):
    with open(output_txt_file, 'w+', encoding='utf-8') as f:
        f.write("Genre prediction explanation\n\n")

        f.write("Original text:\n")
        f.write("{}\n\n".format(original_text))

        f.write("Filtered text:\n")
        f.write("{}\n\n".format(filtered_text))

        f.write("Predicted labels:\n")
        for label in predicted_labels:
            f.write("- {}\n".format(label))
        f.write("\n")

        for label in predicted_labels:
            f.write("Genre: {}\n".format(label))
            f.write("Probability: {:.3f}\n".format(genre_probabilities[label]))
            f.write("Top words:\n")
            for word, weight in top_words_per_label[label].items():
                f.write("- {}: {:.3f}\n".format(word, weight))
            f.write("\n")


def explain_genre_predictions_html(original_text, filtered_text, predicted_labels, genre_probabilities,
                                   top_words_per_label, output_html_file):
    with open(output_html_file, 'w', encoding='utf-8') as f:
        f.write("<html>")
        f.write("<head><title>Genre prediction explanation</title>")
        f.write("</head>")
        f.write("<body>")
        f.write(f"<h1>Here is why the answer is probably {', '.join(predicted_labels)}</h1>")

        f.write("<h2>Original text:</h2>")
        f.write("<p>{}</p>".format(original_text))

        f.write("<h2>Filtered and lemmatized text:</h2>")
        f.write("<p>{}</p>".format(filtered_text))

        f.write("<h2>Predicted labels:</h2>")
        f.write("<ul>")
        for label in predicted_labels:
            f.write("<li>{}</li>".format(label))
        f.write("</ul>")

        for label in predicted_labels:
            f.write("<h2>Genre: {}</h2>".format(label))
            f.write("<p style='color: #009933;'>Probability: {:.3f}</p>".format(genre_probabilities[label]))
            f.write("<h3>Top words:</h3>")
            f.write("<ul>")
            for word, weight in top_words_per_label[label].items():
                f.write("<li>{}: {:.3f}</li>".format(word, weight))
            f.write("</ul>")

        f.write("</body>")
        f.write("</html")


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


def classify_tv_show(input_file, output_json_file, encoding='UTF-8', explanation_output_dir=None):
    try:
        print('Reading the input file: ', input_file)
        original_description = read_file(input_file, encoding)

        multilabel_binarizer_path = f'{paths.location_of_model}/{paths.binarizer_filename}'
        bow_vectorizer_path = f'{paths.location_of_model}/{paths.vectorizer_filename}'
        model_path = f'{paths.location_of_model}/{paths.model_filename}'

        print('Loading saved models: ', [multilabel_binarizer_path, model_path, bow_vectorizer_path])

        multilabel_binarizer = joblib.load(multilabel_binarizer_path)
        bow_vectorizer = joblib.load(bow_vectorizer_path)
        model = joblib.load(model_path)

        filtered_description = filter_description(original_description)
        filtered_description = stopwords_filtering(filtered_description)
        filtered_description = lemmatization_description(filtered_description)

        predicted_labels = multilabel_binarizer.inverse_transform(
            model.predict(bow_vectorizer.transform([filtered_description])))

        all_genre_labels = multilabel_binarizer.classes_

        predicted_labels_list = []
        for labels_series in predicted_labels:
            for label in labels_series:
                predicted_labels_list.append(label)

        genre_probabilities_dict = {}
        for labels in predicted_labels:
            genre_probabilities_dict[", ".join(labels)] = \
                model.predict_proba(bow_vectorizer.transform([filtered_description]))[0]

        top_genres_probs = {}
        for labels, probabilities in genre_probabilities_dict.items():
            prob_dict = dict(zip(all_genre_labels, probabilities))
            top_genres_probs.update(
                {k: v for k, v in sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)})

        feature_names = bow_vectorizer.get_feature_names_out()

        top_words_per_label = {}

        for label_of_interest in predicted_labels_list:
            label_idx = list(all_genre_labels).index(label_of_interest)

            label_weights = model.estimators_[label_idx].coef_

            strengths = pd.Series(index=feature_names, data=label_weights[0])

            top_words = strengths.nlargest(10)

            top_words_per_label[label_of_interest] = top_words

        with open(output_json_file, 'w+', encoding='UTF-8') as json_file:
            json.dump(predicted_labels_list, json_file, ensure_ascii=False)

        print("Output was saved: ", output_json_file)

        if explanation_output_dir:
            location_of_explanation_txt = os.path.join(explanation_output_dir, paths.location_of_explanation_txt)
            location_of_explanation_html = os.path.join(explanation_output_dir, paths.location_of_explanation_html)

            explain_genre_predictions_html(original_description, filtered_description, predicted_labels_list,
                                           top_genres_probs,
                                           top_words_per_label, location_of_explanation_html)
            explain_genre_predictions_txt(original_description, filtered_description, predicted_labels_list,
                                          top_genres_probs,
                                          top_words_per_label, location_of_explanation_txt)

            print("Explanations were saved: ", [location_of_explanation_html, location_of_explanation_txt])

        print(f"Finished classifying...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify TV show genres based on description")
    parser.add_argument("--input-file", required=True, help="Path to the input file with TV show description")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for genres")
    parser.add_argument("--encoding", default="UTF-8", help="Input file encoding (default: UTF-8)")
    parser.add_argument("--explanation-output-dir", help="Directory for explanation output")

    args = parser.parse_args()

    classify_tv_show(args.input_file, args.output_json_file, args.encoding, args.explanation_output_dir)
