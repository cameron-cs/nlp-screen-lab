# Beginning
The `index.py` and `search.py` files are Python scripts for indexing and searching TV shows based on descriptions using GloVE embeddings and cosine similarity. 

## Search engine structure:
- the `glove` directory:
- the directory for storing a `glove.6B.300d.txt` file (embeddings)
- the directory `index` for storing a `indexed_tv_shows.parquet` file (indexed TV shows)
- the directory `model` for storing Word2Vec models files (`glove_model.model`, `glove_model.model.vectors.npy`)

## Prerequisites of the search engine
### Pre-trained word embeddings

Before we delve into the indexing and searching processes, it's essential to highlight the use of pre-trained word embeddings, specifically the `glove.6B.300d.txt` file. 
These embeddings are used to represent words as dense vectors, capturing the semantic meaning of words and phrases.
I haven't included the pre-trained word embeddings to zip file because they are huge (almost 1GB).
To utilize pre-trained embeddings, the following steps are taken:

- **Download GloVE embeddings:** initially, the `glove.6B.300d.txt` file, containing pre-trained GloVE embeddings, is downloaded from a reliable source. This file contains word vectors in a high-dimensional space.

- **The link for the GloVE embeddings**: the `glove.6B.300d.txt` file can be downloaded from the link [https://nlp.stanford.edu/data/glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)

- **Moving the dowloaded file to project dir**: move the downloaded `glove.6B.300d.txt` file to the [glove](glove) directory

## Components of the search engine

### 1. Indexing (`index.py`)

- **Data retrieval:** the search engine begins by loading data from an SQLite database file, which contains information about TV shows, including their descriptions.

- **Data preprocessing:** the loaded data undergoes a series of preprocessing steps to ensure data quality:
  - duplicate rows are removed based on the `tvmaze_id` and `description` columns.
  - rows with missing values in the `tvmaze_id` and `description` columns are removed.
  - noisy descriptions, defined as descriptions with fewer than 10 characters, are filtered out.
  - HTML tags in the descriptions are stripped using BeautifulSoup.

- **Text preprocessing:** after cleaning the descriptions, the text data goes through the following text preprocessing steps:
  - removal of non-alphabetic characters and symbols.
  - tokenization of the descriptions.
  - stopword filtering to remove common words that don't carry significant meaning.

- **Word embeddings:** GloVE word embeddings are used to represent words as vectors. To facilitate this, the raw GloVE embeddings are transferred to Word2Vec format.
  - it is going to be done by the `glove_util.py` right before creating a GloVE Word2Vec model

- **Description encoding:** each TV show description is encoded as a dense vector by averaging the word vectors of the words in the description. If a word in the description is not present in the GloVE model, it is represented as a zero vector.

- **Index creation:** the encoded descriptions, along with `tvmaze_id` and `showname`, are stored in a Pandas DataFrame, which serves as the search index. This index is saved as a Parquet file (the `indexed_tv_shows.parquet` file after indexing)

- **Model saving:** the GloVE model is saved for later use in the search process (the `glove_model.model` and `glove_model.model.vectors.npy` files after indexing)

##### Usage:
```commandline
python index.py --training-data tvmaze.sqlite
```

### 2. Searching (`search.py`)

- **Query qrocessing:** the search engine starts by reading a user's query from an input file. The query undergoes similar text preprocessing as during indexing, including lemmatization, symbol removal, and stopword filtering.

- **Loading index and Word2Vec GloVE model:** the search engine loads the previously created index and the saved GloVE model.

- **Query encoding:** the preprocessed query is encoded into a dense vector using the GloVE model. This vector represents the user's query.

- **Cosine similarity:** cosine similarity is used to measure the similarity between the query vector and the encoded descriptions in the index. A higher cosine similarity indicates a closer match between the query and a TV show description.

- **Matching and sorting:** TV shows are ranked based on their similarity to the query. The search engine sorts the TV shows by descending similarity score.

- **Top matches:** the search engine selects the top matching TV shows, typically the top three based on the highest similarity scores.

- **Output:** the matched TV shows, including their `tvmaze_id` and `showname`, are saved in a JSON file as the search results to the output path (from the cmd line args).

##### Usage:
```commandline
python search.py --input-file input_search.txt --output-json-file results.json
```

## Mathematical concepts

1. **Cosine similarity:**

   - cosine similarity is a mathematical measure used to quantify the similarity between two non-zero vectors in an inner product space.

   - for two vectors, A and B, cosine similarity is defined as the cosine of the angle θ between them.

   - the formula for cosine similarity is:
   
     ```
     cosine_similarity(A, B) = (A ⋅ B) / (||A|| * ||B||)
     ```
     
     where:
     - `A ⋅ B` is the dot product of vectors A and B.
     - `||A||` and `||B||` are the Euclidean norms (magnitudes) of vectors A and B.

   - cosine similarity ranges from -1 (completely dissimilar) to 1 (perfectly similar). A higher cosine similarity indicates a higher degree of similarity between two vectors.

2. **Vector averaging:**

   - in the search engine, TV show descriptions are encoded as dense vectors by averaging the word vectors of the words in the description.

   - the formula for averaging vectors is:

     ```
     Average vector = (Vector1 + Vector2 + ... + VectorN) / N
     ```
     
     where:
     - `Vector1`, `Vector2`, ..., `VectorN` are the word vectors of individual words in the description.
     - `N` is the number of words in the description.

### Explanation

- when the search engine indexes TV show descriptions, it first preprocesses and tokenizes the text, converting each word to its GloVE word embedding vector if available in the model. If a word is not in the model, it is represented as a zero vector.

- the TV show description is encoded as a dense vector by averaging these word vectors. This encoding captures the semantic information of the description in a compact vector form.

- during a search, the user's query is preprocessed and encoded in a similar way to create a query vector.

- the search engine calculates the cosine similarity between the query vector and the encoded TV show descriptions in the index. This similarity measure helps identify which TV show descriptions are most similar to the user's query.

- by ranking TV show descriptions based on their cosine similarity scores, the search engine can provide a list of top matching TV shows that are most relevant to the user's query.

In mathematical terms, this search engine leverages cosine similarity and vector averaging to find TV shows whose descriptions are closest in meaning to the user's query. The higher the cosine similarity between the query and a TV show description, the more similar they are in terms of the words and phrases used. This allows for accurate and effective content recommendations based on the semantic similarity of descriptions.

## Usage of the search engine

To use this search engine, you run `index.py` to create the index of TV show descriptions, and then you can run `search.py` with a query to find the most relevant TV shows based on that query. The search results are saved in a JSON file.

This search engine is designed to efficiently find TV shows based on the similarity of their descriptions to a user's query, enabling content discovery and recommendation. The use of word embeddings and cosine similarity allows for a flexible and powerful search process.

```commandline
python index.py --raw-data tvmaze.sqlite
```

```commandline
python search.py --input-file input_search.txt --output-json-file results.json
```
