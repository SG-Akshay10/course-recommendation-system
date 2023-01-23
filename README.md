# Course Recommendation System

The college course-recommendation-system is a Python-based project that utilizes the NumPy and pandas libraries to recommend college courses to students. The program uses the cosine similarity,sigmoid kernel and decision tree classifer module of the scikit-learn library to determine the similarity between courses based on their course descriptions.

The program takes in a dataset of courses and students' course history as input, performs preprocessing, cleaning and vectorizing of data using TfidfVectorizer, and generates recommendations for courses that are similar to the courses that the student has previously taken. Cosine similarity is then calculated between the student's previously taken courses and all other courses in the dataset.

This project also allows for the input of a student's preferred department, which is used to filter the recommendations to only include courses from that department.

Additionally, the project provides a user-friendly interface through which students can easily input their course history and preferred department. The results are then presented in a clear and organized manner, making it easy for the student to understand and choose the most relevant courses for them.

The project is built using Python and the use of NumPy and pandas libraries allows for efficient data manipulation and processing. The implementation of the cosine similarity module of scikit-learn library ensures accurate and reliable recommendations.

Overall, the college course-recommendation-system project is a valuable tool for students to discover new and relevant courses that align with their interests and academic goals. It can also help educational institutions to provide tailored education for their students. The project is open-source and can be easily integrated into any existing course management system.

# TFIDF Vectorizer

TF-IDF (Term Frequency-Inverse Document Frequency) is a way to represent the importance of a word in a document or a collection of documents (corpus). It is commonly used in natural language processing and information retrieval tasks, such as text classification and text clustering.

The TF-IDF score of a word is calculated by multiplying its term frequency (TF) by its inverse document frequency (IDF). The TF component represents how often a word appears in a document, while the IDF component represents how rare the word is across all documents.

The TfidfVectorizer is a class in the scikit-learn library that can be used to convert a collection of raw documents into a numerical feature matrix, where each row represents a document and each column represents a term. The TfidfVectorizer class automatically calculates the TF-IDF score for each term in the corpus and returns a sparse matrix of the results.

It's important to note that the TfidfVectorizer first performs a tokenization step, to split the input text into words or tokens, and then it computes the tf-idf values for the terms. It's also possible to use the TfidfVectorizer to transform single documents into a feature vector, this is the case of the college course-recommendation-system where it vectorizes the course descriptions.

The TfidfVectorizer provides several parameters to adjust the tokenization process, such as the type of tokenizer, the minimum and maximum number of words to include in the vocabulary, and whether to remove stop words or not.

# Cosine Similarity 

Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. It is a normalized dot product of two vectors and ranges from -1 to 1. The value of cosine similarity is 1 when the vectors point in the same direction, 0 when the vectors are orthogonal (perpendicular) and -1 when the vectors point in the opposite direction.

In the case of the college course-recommendation-system, the vectors in question are the course descriptions. The program uses the TfidfVectorizer method to convert the text descriptions of the courses into numerical vectors. The cosine similarity is then calculated between the student's previously taken courses and all other courses in the dataset, to determine how similar the courses are to each other.

One of the advantages of using cosine similarity is that it is relatively easy to compute, and it is not affected by the magnitude of the vectors, only by their orientation. This makes it particularly useful for comparing texts or documents, as it can ignore the effects of different document lengths.

It's important to note that cosine similarity is a measure of similarity, not dissimilarity, so to obtain the dissimilarity between two vectors is necessary to subtract cosine similarity from 1.
