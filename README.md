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

# Sigmoid Function 

The sigmoid function is a mathematical function that maps any input value to a value between 0 and 1. It is often used in machine learning and statistics as a way to model the probability of an event or a binary outcome. The sigmoid function is defined as:

f(x) = 1 / (1 + e^(-x))

Where e is the base of the natural logarithm, and x is the input value.

The shape of the sigmoid function is a "S" shaped curve, which is why it is also referred to as the "Sigmoid curve". It has an "S" shape because when the input values are very small (i.e., x << 0), the function outputs values close to 0, and when the input values are very large (i.e., x >> 0), the function outputs values close to 1.

The sigmoid function is commonly used in logistic regression, a type of supervised learning algorithm that is used for classification tasks. In logistic regression, the sigmoid function is used as the activation function in the output layer of the model. It maps the output of the linear equation of the input features to a probability value between 0 and 1, which can then be thresholded to make predictions.

It's also common to use sigmoid function in neural networks, for example, in the output layer of a binary classification problem, where the sigmoid function can be used to map the output of the last layer to a probability of belonging to a particular class.

It's important to note that as the input values approach to infinity, the sigmoid function returns 1, and as the input values approach to negative infinity, the sigmoid function returns 0.

# Decision Tree Classifer 

A Decision Tree Classifier is a type of supervised learning algorithm that is used for classification problems. It works by creating a tree-like model of decisions and their possible consequences. Each internal node in the tree represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label. The goal of the decision tree classifier is to split the data into smaller subsets that are as homogeneous as possible with respect to the class label.

The decision tree classifier starts at the root node, which represents the entire dataset, and splits it into subsets based on the values of the input features. The algorithm then recursively repeats this process for each subset, creating a new internal node for each test and new branches for each outcome. The process stops when a stopping criterion is met, for example, when all the instances in a subset belong to the same class or when there are no more features to split on.

The decision tree classifier uses a variety of techniques to determine the best feature and the best threshold value to split the data on. Common techniques include information gain, Gini index, and gain ratio. These techniques measure the impurity of the data before and after the split and the goal is to maximize the decrease in impurity, which in turn increases the homogeneity of the subsets.

Once the decision tree is built, it can be used to make predictions on new instances by traversing the tree from the root to a leaf node. The class label of the leaf node is the prediction of the decision tree classifier.

Decision tree classifiers are popular because they are easy to interpret and understand, and they can handle both numerical and categorical data. They also perform well on large datasets and can handle missing data and outliers. However, they can be prone to overfitting, especially when the tree is deep and the data is noisy. To avoid overfitting, techniques such as pruning or limiting the maximum depth of the tree can be used.
