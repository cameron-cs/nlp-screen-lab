# nlp-screen-lab
Advanced NLP methodologies to analyze and understand the textual descriptions of TV shows

This project, a sophisticated blend of artificial intelligence techniques applied to the domain of TV shows, consists of two main components: a genre classifier and a semantic search engine. Both parts leverage advanced natural language processing (NLP) methodologies to analyze and understand the textual descriptions of TV shows, facilitating both genre classification and content-based search functionality. Here’s a detailed overview of each component:

## Genre classifier
The genre classifier is designed to predict the genre(s) of TV shows based on their descriptions. It operates through a pipeline beginning with data preparation, where TV show descriptions and genres are extracted from a database. This data undergoes extensive preprocessing, including tokenization, lemmatization, and removal of stopwords and non-alphabetic characters, to ensure quality and consistency.

Feature extraction is performed using the Bag of Words (BoW) model, transforming the cleaned descriptions into a numerical format that a machine learning model can process. The classifier employs logistic regression, a robust algorithm suitable for multi-label classification, to predict the genre(s) based on these numerical features. The model’s performance is evaluated through accuracy metrics, ensuring the predictions are both reliable and relevant.

The classifier is supported by a structured pipeline consisting of training (train.py) and classification (classify.py) scripts, along with necessary directories for storing the model files (model) and explanations (explanation), addressing the need for interpretability.

## Semantic search engine
The semantic search engine component provides the capability to find TV shows closely matching user queries, based on the semantic similarity of their descriptions. This functionality is enabled through the use of GloVe (Global Vectors for Word Representation) embeddings, which represent words as dense vectors, capturing their semantic meanings.

The search engine process begins with indexing (index.py), where TV show descriptions are preprocessed and converted into vector representations by averaging the GloVe embeddings of the words they contain. These vectors, along with metadata like TV show names and IDs, are stored in an index for efficient retrieval.

During the search phase (search.py), user queries undergo a similar preprocessing and vectorization process. The system then calculates the cosine similarity between the query vector and the vectors in the index to find the most semantically similar TV show descriptions. TV shows are ranked by similarity, and the top matches are presented to the user.

This project demonstrates a comprehensive application of NLP and AI techniques to a practical problem, showcasing skills in text preprocessing, machine learning, vector space models, and similarity measures. The blend of a genre classifier and a semantic search engine provides a holistic approach to navigating and understanding TV show content through textual descriptions, offering significant value to platforms seeking to enhance content discoverability and recommendation systems.