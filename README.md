# French Text Difficulty Classification Task 🥐

## Team Approach and Ranking

For this project, we ensured an equitable division of labor. Initially, we worked separately to identify models that could be effective for our project. As the deadline approached, we collaborated more closely to integrate the best elements each of us had discovered in the models, including parameters, model structures, data augmentation techniques, and more. For the GitHub repository, both of us contributed content and reviewed each other's work for double verification. Regarding the YouTube video, we filmed it together, Alexandra handled the editing, while Theodore focused on the user interface. Finally, we really enjoyed working together for this project ! ❤️

After 84 submissions, we finished eleventh out of twenty in the Kaggle competition.

## Description 
Effectively learning a new language requires reading texts that match the learner's proficiency level. For non-French speakers, finding French texts that align with their skills (from A1 to C2 levels) can be challenging. To address this, we have developed a model that predicts the difficulty of French-written texts based on the Common European Framework of Reference for Languages (CEFR).

Consistent reading of appropriately leveled texts is key to improving language skills. By matching texts to a learner's level, they encounter familiar words while being introduced to new vocabulary, striking a crucial balance for effective learning and progression.

### What is CEFR? 📘
The Common European Framework of Reference for Languages (CEFR) is an internationally recognized standard for describing language ability. It is widely used to assess and describe the language proficiency of learners, making it a reliable benchmark for our text difficulty predictions.

By predicting the CEFR level of French texts, our project aims to facilitate personalized language learning experiences, enhancing the effectiveness of reading as a tool for language acquisition.

More information on the CEFR can be found [here](https://www.coe.int/en/web/common-european-framework-reference-languages/level-descriptions).

## Repository Index 🗂️

- `README.md`: This file, providing an overview for the project.
- `Data/`: Directory containing the datasets.
  - `training_data.csv`: Training data with French texts and their corresponding CEFR levels.
  - `unlabelled_test_data.csv`: Test data used for model evaluation.
  - `sample_submission.csv`: Submission sample for Kaggle
  - `new_attempt.csv`: Last submission on Kaggle
  - `flaubert_difficulty_predictions.csv`: A submission on Kaggle
- `Notebooks/`: Directory containing Jupyter notebooks for data analysis and model training.
  - `exploratory_analysis.ipynb`: Notebook for initial data exploration and visualization.
  - `model_experiments_minimal_preprocessing.ipynb`: Notebook for experimenting with different models, with minimal data preprocessing and no hyper-parameter tuning (Logistic Regression, SVC, Random Forest, Decision Tree, KNN).
  - `model_experiments_best.ipynb`: Notebook for experimenting with different models with more data preprocessing and hyper-parameter tuning (Logistic Regression, SVC, Random Forest, Decision Tree, KNN).
  - `Word_Swapping.ipynb`: Notebook for the word swapping.
  - `Synonym_Remplacement.ipynb`: Notebook for the synonym remplacement.
  - `OpenAI.ipynb`: Notebook for the OpenAI augmentation.
  - `FlauBERT.ipynb`: Notebook for the FlauBert code.
  - `CamemBERT.ipynb`: Notebook for the CamemBert code.
  - `Two_Step.ipynb`: Notebook for the Two-Step code.
- `Images/`: Directory containing images used in the project documentation.
- `requirements.txt`: List of Python packages required to run the project.

## Table of Contents
- [Exploratory Data Analysis (EDA) 📊](#exploratory-data-analysis-eda-)
- [Initial Models 🍼](#initial-models-)
- [LSTM 🤖](#lstm-)
- [Natural Language Processing (NLP) Models 🗣️](#natural-language-processing-nlp-models-️)
- [CamemBERT Two-Step Classification ✌️](#camembert-two-step-classification-️)
- [CamemBERT Model 🧀](#camembert-model-)
- [FlauBert Model 👨🏻‍🦳](#flaubert-model-)
- [Data Augmentation 📈](#data-augmentation-)
- [User Interface 💻](#user-interface-)
- [Our Video 📹](#our-video-)
  
## Exploratory Data Analysis (EDA) 📊

To begin our analysis, let's first describe our datasets:

### Training Dataset
The training dataset contains 4,800 lines of French text, each labeled with a difficulty level. It includes the following features:
- **id**: A unique identifier for each sentence.
- **sentence**: The French text sentence.
- **difficulty**: The CEFR difficulty level of the sentence.

### Test Dataset
The test dataset contains 1,200 lines of unlabeled French text. It includes the following features:
- **id**: A unique identifier for each sentence.
- **sentence**: The French text sentence.

### Sample from the Training Dataset
Here is a sample from the training dataset, showing a sentence and its corresponding difficulty level:

| id  | sentence                                     | difficulty |
|-----|----------------------------------------------|------------|
| 10  | Bonjour et bonne année.                      | A1         |
| 39  | Tu mangeas les petits fruits dès que tu les eus cueillis  | C2  |
| 76  | Vous ferez de la randonnée ?                 | B2 |

The sample above illustrates the type of data used for training our model. Each sentence is paired with a difficulty level, which helps the model learn to predict the appropriate CEFR level for new sentences.

After examining the structure of our datasets, we delve deeper into analyzing the training data to understand the complexity associated with different difficulty levels. Below is a summary table showing the average number of word in a sentence and the count of sentences per difficulty level in the training dataset:

| Difficulty | Mean Length | Count |
|------------|-------------|-------|
| A1         | 7.442804    | 813   |
| A2         | 11.601258   | 795   |
| B1         | 14.979874   | 795   |
| B2         | 19.133838   | 792   |
| C1         | 24.357143   | 798   |
| C2         | 31.517968   | 807   |

The table reveals a clear trend: as the difficulty level increases from A1 to C2, so does the average number of word in a sentence. This suggests that higher difficulty texts involve more complex vocabulary. Longer sentence typically indicate advanced linguistic structures such as compound words, specialized terminology, and sophisticated conjugations. These elements contribute to higher cognitive load and lexical richness, which are expected in texts classified at higher CEFR levels.

### Visualization of Sentence Length Distribution by Difficulty Level

We created a boxplot to visually explore how sentence length varies across different CEFR difficulty levels. This boxplot demonstrates the trend that as difficulty increases, the length of sentences tends to increase as well. This suggests that more complex sentences, often involving more clauses and concepts, are categorized into higher difficulty levels.

![Boxplot of Sentence Length by Difficulty Level](Images/sentence_length_distribution.png)

### Analysis of Average Word Length by Difficulty Level

Furthermore, we analyzed the average word length within sentences across different difficulty levels using a kernel density estimation (KDE) plot. This plot helps illustrate that not only the length of sentences but also the complexity of words used tends to increase with higher difficulty levels. Longer words often indicate more complex vocabulary or specialized terms, which are more frequent in advanced texts.

![KDE Plot of Average Word Length by Difficulty Level](Images/word_length_distribution.png)

### Conclusion

The visual and statistical analyses presented above provide clear evidence of how textual complexity in terms of sentence and word length correlates with the assigned CEFR difficulty levels. Longer sentences and words are indicative of higher difficulty levels, reflecting the increased linguistic complexity required at each subsequent level. Understanding these patterns is essential for developing an effective model to predict the difficulty level of unseen French texts based on their linguistic features.

All the necessary codes can be found in the following notebook:
- [`Notebooks/Data_Exploration_Visualization.ipynb`](Notebooks/Data_Exploration_Visualization.ipynb)

## Initial Models 🍼

### Introduction

We start by experimenting with some basic machine learning models that we are familiar with. The goal is to establish a baseline performance before gradually incorporating more advanced pre-processing techniques to improve the model accuracy.

### Model Descriptions

#### **Logistic Regression**
Logistic Regression is a linear model used for classification problems. It estimates the probability that a given input belongs to a certain class using a logistic function. It is particularly useful for binary classification but can be extended to multi-class problems. For more details, see [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).

#### **Support Vector Classifier (SVC)**
Support Vector Classifier (SVC) is a powerful classification algorithm that works by finding the hyperplane that best separates the classes in the feature space. Different kernels can be used to transform the input data to find an optimal boundary. For more details, visit [Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

#### **Random Forests**
Random Forests are ensemble learning methods that operate by constructing multiple decision trees during training and outputting the mode of the classes for classification. They help to improve predictive accuracy and control overfitting. For more details, see [Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#forest).

#### **Decision Trees**
Decision Trees are non-parametric supervised learning methods used for classification and regression. They create a model that predicts the value of a target variable by learning simple decision rules from the data features. For more details, see [Decision Trees](https://scikit-learn.org/stable/modules/tree.html).

#### **K-Nearest Neighbors (KNN)**
K-Nearest Neighbors is a simple, instance-based learning algorithm that classifies a data point based on how its neighbors are classified. It is a type of lazy learning where the function is only approximated locally and all computation is deferred until function evaluation. For more details, see [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#classification).

### Minimal Data Processing

All the necessary libraries can be found in the following notebook:
- [`requirements.txt`](requirements.txt)

1. **Label Encoding**: We encode the categorical difficulty labels (A1, A2, B1, B2, C1, C2) into numerical values using `LabelEncoder`.

    ```python
    from sklearn.preprocessing import LabelEncoder

    # Encode the labels
    label_encoder = LabelEncoder()
    df_training_data['difficulty_encoded'] = label_encoder.fit_transform(df_training_data['difficulty'])
    ```

2. **Vectorization**: We convert the text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Vectorize the sentences
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(df_training_data['sentence'])
    ```

3. **Train-Test Split**: We split the data into training and validation sets using an 80-20 split.

    ```python
    from sklearn.model_selection import train_test_split

    # Split the data into training, validation, and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_vectorized, df_training_data['difficulty_encoded'], test_size=0.2, random_state=42)
    ```

### Model Training

After preparing our data with minimal preprocessing, we proceed to create and train our models. For each model, we train the model on the training data, and evaluate its performance on the validation data. We record the evaluation metrics, which include accuracy, precision, recall, and F1-score. These metrics help us understand how well the model performs in predicting the difficulty levels of the French text. The full code for all models can be found in the following notebook:

- [`Notebooks/model_experiments_minimal_preprocessing.ipynb`](Notebooks/model_experiments_minimal_preprocessing.ipynb)

### Evaluation Metrics

We evaluate the performance of each model using the following metrics:

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **F1-Score**: The weighted average of Precision and Recall.

Here is the table for recording the performance metrics of each model:


| Metric       | **Logistic Regression** | **SVC** | **Random Forests** | **Decision Tree** | **KNN** |
|--------------|-------------------------|---------|---------------------|-------------------|---------|
| Accuracy     | 0.4490                  | 0.4542  | 0.4052              | 0.3260            | 0.3198  |
| Precision    | 0.4410                  | 0.4495  | 0.4020              | 0.3224            | 0.3950  |
| Recall       | 0.4490                  | 0.4542  | 0.4052              | 0.3260            | 0.3198  |
| F1-Score     | 0.4400                  | 0.4499  | 0.3904              | 0.3229            | 0.2956  |


### Conclusion

Based on the initial experiments with minimal preprocessing, we observe the following performance metrics for each model:

- **Logistic Regression**: Achieved an accuracy of 0.4490, with balanced precision and recall, indicating a moderate performance.
- **SVC**: Slightly outperformed Logistic Regression with an accuracy of 0.4542, making it the best performing model in this initial phase.
- **Random Forests**: Had a lower accuracy of 0.4052, suggesting that it may benefit from further tuning or more advanced preprocessing.
- **Decision Tree**: Showed even lower accuracy at 0.3260, indicating a tendency to overfit without proper pruning or tuning.
- **KNN**: Had the lowest accuracy at 0.3198, which might be improved by adjusting the number of neighbors or scaling the data.

Overall, SVC demonstrated the best performance among the models tested. However, all models show room for improvement. Future steps will include more extensive preprocessing, hyperparameter tuning, and possibly exploring more advanced algorithms to enhance prediction accuracy.

### Initial Models with More Pre-Processing and Hyper-Parameter Tuning

#### Enhanced Pre-Processing Steps

In this section, we introduce additional pre-processing steps to improve the performance of our models. Below are the detailed steps and their purposes:

1. **Text Cleaning**:
   We define a function `clean_text_french` to normalize and clean the text data. This involves:
   - **Normalization**: Converting characters to their canonical form using `unicodedata.normalize`.
   - **Lowercasing**: Converting all characters to lowercase.
   - **Removing Punctuation**: Removing all punctuation characters.
   - **Removing Digits**: Removing all numerical digits.
   - **Whitespace Reduction**: Reducing multiple spaces to a single space and stripping leading/trailing spaces.

   ```python
   def clean_text_french(text):
       text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
       text = text.lower()
       text = re.sub(r'[^\w\s]', ' ', text)
       text = re.sub(r'\d+', '', text)
       text = re.sub(r'\s+', ' ', text)
       text = text.strip()
       return text
   ```


2. **Text Cleaning Application**:
Applying the text cleaning function to the sentences in both training and test datasets.
```python
df_training_data['sentence'] = df_training_data['sentence'].apply(clean_text_french)
df_unlabelled_test_data['sentence'] = df_unlabelled_test_data['sentence'].apply(clean_text_french)
```
3. **Tokenization**:
Tokenizing the cleaned sentences to prepare them for Word2Vec training.

```python
nltk.download('punkt')
df_training_data['tokens'] = df_training_data['sentence'].apply(word_tokenize)
df_unlabelled_test_data['tokens'] = df_unlabelled_test_data['sentence'].apply(word_tokenize)
```

4. **Word2Vec Training**:
Training a Word2Vec model on the combined tokenized datasets to capture semantic relationships between words.

```python
combined_tokens = pd.concat([df_training_data['tokens'], df_unlabelled_test_data['tokens']])
model_w2v = Word2Vec(combined_tokens, vector_size=100, window=5, min_count=1, workers=4)
```

5. **Sentence Vectorization**:
Defining a function sentence_to_vector to convert tokenized sentences into vectors using the trained Word2Vec model. This function computes the mean of the word vectors for each token in the sentence.

```python
def sentence_to_vector(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
```

6. **Vectorizing Data with Word2Vec**:
Converting the tokenized sentences into vectors for both training and test datasets using the Word2Vec model.

```python
w2v_features_train = np.array([sentence_to_vector(tokens, model_w2v) for tokens in df_training_data['tokens']])
w2v_features_test = np.array([sentence_to_vector(tokens, model_w2v) for tokens in df_unlabelled_test_data['tokens']])
```

7. **TF-IDF Vectorization**:
Applying TF-IDF vectorization to the cleaned sentences to capture the importance of words in the context of the dataset.

```python
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(df_training_data['sentence'])
X_test_tfidf = vectorizer.transform(df_unlabelled_test_data['sentence'])
```

8. **Combining Features**:
Combining the TF-IDF features and Word2Vec features to form a comprehensive feature set for training and test datasets.

```python
X_train_combined = np.hstack((X_train_tfidf.toarray(), w2v_features_train))
X_test_combined = np.hstack((X_test_tfidf.toarray(), w2v_features_test))
```

9. **Label Encoding**:
Encoding the difficulty labels using LabelEncoder.

```python
label_encoder = LabelEncoder()
df_training_data['difficulty_encoded'] = label_encoder.fit_transform(df_training_data['difficulty'])
y = df_training_data['difficulty_encoded']
```

10. **Train-Test Split**:
Splitting the training data into training and validation sets.

```python
X_train, X_val, y_train, y_val = train_test_split(X_train_combined, y, test_size=0.2, random_state=42)
```

### Model Training with Enhanced Pre-Processing
After applying the enhanced pre-processing steps, we re-run all the models: Logistic Regression, Support Vector Classifier, K-Nearest Neighbors (KNN), Decision Tree, and Random Forest. Below are the results of the model evaluation metrics.


| Metric       | **Logistic Regression** | **SVC** | **Random Forests** | **Decision Tree** | **KNN** |
|--------------|-------------------------|---------|---------------------|-------------------|---------|
| Accuracy     | 0.4563                  | 0.4563  | 0.3844              | 0.3427            | 0.4063  |
| Precision    | 0.4547                  | 0.4540  | 0.3842              | 0.3411            | 0.4070  |
| Recall       | 0.4563                  | 0.4563  | 0.3844              | 0.3427            | 0.4063  |
| F1-Score     | 0.4538                  | 0.4539  | 0.3826              | 0.3415            | 0.3984  |


### Conclusion

By comparing the two sets of model evaluation metrics, we can observe the impact of enhanced pre-processing on the performance of our models:

- **Logistic Regression** and **SVC**: Both models show a slight improvement in accuracy, precision, recall, and F1-score after applying enhanced pre-processing steps.
- **Random Forest**: The accuracy of Random Forest has slightly decreased from 0.4052 to 0.3844. This suggests that the additional pre-processing steps may not have benefited this model as much as the others, possibly due to the nature of the Random Forest algorithm which can handle unprocessed data better.
- **Decision Tree**: There is a noticeable improvement in the accuracy and other metrics for the Decision Tree model, indicating that the enhanced pre-processing steps helped this model perform better.
- **KNN**: The accuracy and F1-score for KNN have improved significantly, showing that this model benefited from the additional pre-processing.


We will now focus on the models with the highest accuracy for hyper-parameter tuning. Specifically, we will tune the hyperparameters for **Logistic Regression** and **SVC** to further improve their performance.

### Hyper-Parameter Tuning for Logistic Regression

To find the optimal hyperparameters for Logistic Regression, we use `RandomizedSearchCV` from scikit-learn. This method performs a random search over specified hyperparameter values and performs cross-validation to find the best set of parameters.

1. **Define the parameter distribution**: We specify a distribution of hyperparameters to search over, including different values of `C`, `penalty`, and `solver`.
2. **Initialize and run RandomizedSearchCV**: We initialize `RandomizedSearchCV` with our Logistic Regression model and the parameter distribution, and then fit it to the training data.
3. **Get the best estimator**: After fitting, `RandomizedSearchCV` provides the best combination of parameters which we can then use to re-evaluate the model.


### Hyper-Parameter Tuning for SVC

Similarly, we tune the hyperparameters for SVC using `RandomizedSearchCV`.

1. **Define the parameter distribution**: We specify a distribution of hyperparameters to search over, including different values of `C`, `kernel`, and `gamma`.
2. **Initialize and run RandomizedSearchCV with StratifiedKFold**:
   - We use `StratifiedKFold` to ensure that the folds are made by preserving the percentage of samples for each class. This helps in maintaining the distribution of classes across the folds.
   - We initialize `RandomizedSearchCV` with our SVC model, the parameter distribution, and `StratifiedKFold`, and then fit it to the training data.
3. **Get the best estimator**: After fitting, `RandomizedSearchCV` provides the best combination of parameters which we can then use to re-evaluate the model.
   
These steps allow us to find the optimal hyperparameters for Logistic Regression and SVC, and re-run the models with the best parameters to improve their accuracy and other performance metrics.

For the complete code for both, refer to the notebook:
- [`Notebooks/model_experiments_best.ipynb`](Notebooks/model_experiments_best.ipynb)

Here is the table with the new results:

#### Updated Table with New Data:

| Metric       | **Logistic Regression** | **Best Logistic Regression** | **SVC** | **Best SVC** | **Random Forests** | **Decision Tree** | **KNN** |
|--------------|-------------------------|------------------------------|---------|--------------|---------------------|-------------------|---------|
| Accuracy     | 0.4563                  | 0.4594                       | 0.4563  | 0.4698       | 0.3844              | 0.3427            | 0.4063  |
| Precision    | 0.4547                  | 0.4582                       | 0.4540  | 0.4664       | 0.3842              | 0.3411            | 0.4070  |
| Recall       | 0.4563                  | 0.4594                       | 0.4563  | 0.4698       | 0.3844              | 0.3427            | 0.4063  |
| F1-Score     | 0.4538                  | 0.4573                       | 0.4539  | 0.4670       | 0.3826              | 0.3415            | 0.3984  |

After performing hyper-parameter tuning, we can observe that the accuracy of our models has improved. The best model among those initially explored is **SVC** with the optimal parameters.

- **Logistic Regression**: The accuracy increased from 0.4563 to 0.4594, with improvements in precision, recall, and F1-score.
- **SVC**: This model showed the most significant improvement, with the accuracy increasing from 0.4563 to 0.4698. Precision, recall, and F1-score also improved, making it the best performing model with the tuned parameters.

### Analyzing the Best SVC Model

To further understand the performance of our best model, the SVC, we examine the confusion matrix and investigate some erroneous predictions. This helps us to identify where the model is making mistakes and gain insights into potential improvements.

#### Analysis of the Confusion Matrix

![Normalized Confusion Matrix](Images/Normalized_Confusion_Matrix.png)

The confusion matrix provides valuable insights into the performance of our best SVC model. It shows how often our classifier correctly predicted each difficulty level versus how often it confused one level with another. Below are a few key observations:

- **Class A1** is frequently confused with **Class B2**. This indicates that the model struggles to differentiate between very easy and slightly more complex texts.
- **Class B2** and **Class B1** have significant overlap, suggesting that intermediate levels are often challenging to distinguish.
- **Class C2** (the most advanced level) is sometimes predicted as **Class B2**, indicating difficulty in identifying the highest level of difficulty correctly.

#### Examples of Erroneous Predictions

Below are some specific examples of erroneous predictions made by the model:

1. **Original text**: "c est la couleur de nombreux fruits et legumes comme les tomates les fraises ou les cerises"  
   **True label**: A1  
   **Predicted label**: B2

2. **Original text**: "les francais ne cedent pas au chacun pour soi mais ils s interessent d abord a leur cercle familial proche"  
   **True label**: B2  
   **Predicted label**: B1

3. **Original text**: "j ai retrouve le plaisir de manger un oeuf a la coque"  
   **True label**: A2  
   **Predicted label**: B1

4. **Original text**: "trop de charlatans et songe creux en partie malhonnetes en partie dupes de leur propre enthousiasme ont fait au genre humain de magnifiques promesses qu ils etaient bien empeches de tenir"  
   **True label**: C2  
   **Predicted label**: B2

5. **Original text**: "je vois trois pommes de terre mais elles ne sont pas bonnes"  
   **True label**: A1  
   **Predicted label**: A2

### Observations

From the examples and the confusion matrix, we observe a few patterns:

1. **Proximity of Levels**: The model often confuses adjacent difficulty levels, such as A1 with A2 or B2 with B1. This indicates that the features distinguishing these levels might not be sufficiently distinct.
2. **Complex Sentences**: Sentences with more complex structure and vocabulary tend to be misclassified as being at a higher difficulty level. For example, an A1 text with more descriptive content is predicted as B2.
3. **Intermediate Levels**: There is a notable challenge in correctly classifying intermediate levels (B1 and B2), likely due to their nuanced differences in language complexity and usage.

These observations suggest areas for potential improvement, such as refining the feature extraction process, incorporating additional linguistic features, or gathering more labeled data to help the model learn more distinct patt

## LSTM 🤖

### Introduction

In our quest to improve the performance of our text difficulty classification model, we explored various advanced approaches. One such approach is using Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN) known for its effectiveness in sequence prediction problems. LSTM networks are particularly well-suited for natural language processing tasks because they can capture long-term dependencies in text, making them an excellent choice for understanding and classifying the complexity of French texts based on the Common European Framework of Reference for Languages (CEFR).

### Why LSTM?

LSTM networks have the ability to remember previous inputs over long sequences, which is crucial for understanding the context in natural language. Unlike traditional RNNs, LSTMs address the vanishing gradient problem, allowing them to learn long-term dependencies more effectively. This makes LSTMs particularly powerful for tasks like text classification, where the context of words and sentences plays a significant role in determining their meaning and difficulty.

### Implementation Details

We implemented an LSTM-based model to classify French texts into their respective CEFR difficulty levels. Here is a detailed breakdown of the implementation process:

1. **Data Preparation**:
   - **Loading Data**: We loaded the dataset containing French sentences and their corresponding CEFR difficulty levels.
   - **Label Encoding**: We encoded the difficulty levels (A1, A2, B1, B2, C1, C2) into numerical values using `LabelEncoder`.
   - **Data Splitting**: The data was split into training and test sets using an 80-20 split.


2. **Text Tokenization and Padding**:
   - **Tokenizer**: We used the `Tokenizer` class from TensorFlow to convert the text into sequences of integers, with a vocabulary size limited to the most frequent 10,000 words.
   
   - **Padding**: The sequences were padded to ensure uniform length, with a maximum length of 100 words per sequence.
    

3. **Model Architecture**:
   - **Embedding Layer**: This layer converts the integer sequences into dense vectors of fixed size (128 dimensions).
   - **Spatial Dropout**: Applied to the embedding layer to prevent overfitting.
   - **LSTM Layer**: An LSTM layer with 64 units, incorporating dropout and recurrent dropout to further prevent overfitting.
   - **Dense Layer**: A fully connected layer with a softmax activation function to output the probability distribution over the six difficulty levels.
     ```python
     model = Sequential()
     model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
     model.add(SpatialDropout1D(0.2))
     model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
     model.add(Dense(len(label_encoder.classes_), activation='softmax'))
     ```

4. **Model Training**:
   - **Compilation**: The model was compiled using the Adam optimizer and sparse categorical cross-entropy loss.
   - **Early Stopping**: Early stopping was implemented to halt training if the validation loss did not improve for three consecutive epochs.
   - **Training**: The model was trained for 20 epochs with a batch size of 32, using 20% of the training data for validation.

5. **Evaluation**:
   - **Testing**: The model was evaluated on the test set, achieving an accuracy of approximately **0.42**.

### Hyperparameter Tuning with Keras Tuner

As we were not satisfied with the accuracy achieved by our initial LSTM model, and to enhance the performance, we decided to employ hyperparameter tuning using Keras Tuner, a library for hyperparameters tuning for Keras models.

1. **Install Keras Tuner**
2. **Data Preparation**: The data preparation steps remained the same as in the initial implementation.
3. **Defining the HyperModel**: We defined a custom LSTMHyperModel class for hyperparameter tuning.
4. **Hyperparameter Tuning**: We configured Keras Tuner. We used the RandomSearch tuner for hyperparameter optimization.

```python
tuner = RandomSearch(
    LSTMHyperModel(num_classes=num_classes),
    objective='val_accuracy',
    max_trials=10,  # Number of different hyperparameter combinations to try
    executions_per_trial=2,  # Number of models to build and fit for each trial
    directory='my_dir',
    project_name='lstm_tuning')
```
5. **Model Training**: Using the best hyperparameters found by Keras Tuner, we trained the model.
6. **Evaluation**: The best model was evaluated on the test set. 

Despite our efforts with hyperparameter tuning, the LSTM model only achieved a slight improvement, reaching a test accuracy of 42.60%. This minimal enhancement prompted us to explore other models that could better handle the complexity of our text classification task.

## Natural Language Processing (NLP) Models 🗣️

As we progressed with our initial models, we noticed that the performance metrics were not meeting our expectations. The accuracy of our  models, even after pre-processing and hyper-parameter tuning, was not as high as we desired. Recognizing that the core of our challenge lies in effectively understanding and classifying the complexity of French texts, we sought models that inherently understand the linguistic nuances of the French language. This is where specialized NLP models like FlauBERT and CamemBERT come into play.

### Why FlauBert and CamemBert?
FlauBERT and CamemBERT are two state-of-the-art NLP models specifically designed for the French language, building on the success of BERT (Bidirectional Encoder Representations from Transformers) developed by Hugging Face. BERT revolutionized the field of NLP by enabling models to understand the context of a word in a sentence by looking at the words that come before and after it. This bidirectional approach allows for a deeper understanding of language compared to previous unidirectional models.

![Unidirectional context vs bidirectional context](Images/Unidirectional_vs_Bidirectional_Context.png)

*<sub>This image compares unidirectional and bidirectional contexts. On the left, the unidirectional context shows how representations are built incrementally. On the right, BERT's bidirectional context shows how words can "see themselves," allowing for a deeper understanding of the context in both directions. This illustrates one of the key advantages of BERT over previous models.</sub>*

## CamemBert Two-Step Classification ✌️

### The Model

- **Training Data**: CamemBERT was trained on the OSCAR corpus, a massive dataset containing texts extracted from the Common Crawl, representing a diverse array of French text from the internet. This comprehensive dataset helps the model generalize well across different types of French text.
- **Architecture**: CamemBERT uses the RoBERTa framework, which optimizes the pre-training process by using dynamic masking and training on longer sequences. This results in a model that is more robust and capable of handling complex linguistic structures.
- **Applications**: CamemBERT excels in various NLP tasks. For our project, its ability to accurately understand and classify French text is crucial for determining the CEFR levels of the texts.

### Methodology

1. **Label Encoding:**
To begin with, we encoded the difficulty levels into two categories: general levels (A, B, C) and detailed levels (A1, A2, B1, B2, C1, C2). This helps in structuring the classification task into a two-step process, where we first classify the text into a general difficulty level, and then further classify it into a more detailed level.

```python
# Encode labels for general levels
label_encoder_general = LabelEncoder()
df['difficulty_level'] = df['difficulty'].apply(lambda x: x[0])  # Convert A1 to A, B1 to B, etc.
df['difficulty_encoded'] = label_encoder_general.fit_transform(df['difficulty_level'])

# Encode labels for detailed levels (A1, A2, B1, B2, C1, C2)
label_encoder_detailed = LabelEncoder()
df['difficulty_detail_encoded'] = label_encoder_detailed.fit_transform(df['difficulty'])
 ```

2. **Data splitting:**
We split our dataset into training and testing sets to evaluate the model's performance effectively. The split ensures that we have a robust evaluation of how well our model generalizes to unseen data.

3. **Tokenization:**
Using the CamemBERT tokenizer, we prepared the input text data by converting the sentences into tokenized format suitable for the model. Tokenization includes truncating and padding sentences to a uniform length and converting them into tensors.

4. **Model Initialization and Training:**
We initialized a CamemBERT model for sequence classification to handle the general classification task (A, B, C). The model was trained using an AdamW optimizer, with the training process including multiple epochs to ensure the model learns effectively from the data.

5. **Detailed Classification:**
After the general classification, we used the predictions from the general model to further classify the text into detailed levels (A1, A2, B1, B2, C1, C2). Separate datasets and tokenizers were prepared for this fine-grained classification, and a new model was trained for each detailed level classification.

### Conclusion

Splitting the classification process into two steps, general and detailed, provides several advantages. First, it simplifies the initial classification task by categorizing the text into broader difficulty levels (A, B, C), making the model more efficient and focused. Once the text is classified into a general category, the second step allows for a more fine-grained classification (A1, A2, B1, B2, C1, C2). This hierarchical approach helps in managing class imbalance and complexity, improving the overall accuracy and robustness of the model. By tackling easier, broader classifications first, the model can better allocate its resources and learning capacity to the more nuanced, detailed classifications, leading to more accurate predictions and a better understanding of the linguistic intricacies in French texts.

Our our validation set, we had more than **0.70** in accuracy for the first step, and an accuracy of **0.66** for the fine classification. However when submitted to Kaggle, our model achieved an accuracy of below **0.6**.

All the necessary codes can be found in the following notebook:
- [`Notebooks/Two_Step.ipynb`](Notebooks/Two_Step.ipynb)

## CamemBert Model 🧀 
_Best Model_ 🥇

### Methodology

1. **Data Preparation:**
   - We loaded and encoded the training data, transforming categorical difficulty levels into numerical values suitable for machine learning.
   
2. **Handling Class Imbalance:**
   - Class weights were computed to address class imbalance, ensuring the model pays adequate attention to less frequent classes during training.

```python
# Compute class weights to handle imbalance
class_wts = compute_class_weight('balanced', classes=np.unique(df_train['difficulty_encoded']), y=df_train['difficulty_encoded'])
class_wts_tensor = torch.tensor(class_wts, dtype=torch.float).to(device)

# Define the loss function with class weights
loss_function = CrossEntropyLoss(weight=class_wts_tensor)
```

3. **Custom Dataset Class:**
   - A custom dataset class was implemented to handle the tokenization and preparation of text data using the CamemBERT tokenizer.

4. **Model Training:**

   - The CamemBERT model was trained on the training dataset. 
   - For the number of epochs and learning rate, we experimented multiple times to find the best combination. The chosen values provided the best results in terms of model performance and stability.
   - During each training iteration, gradients are computed, model parameters are updated, and old gradients are cleared. This process ensures effective learning by making appropriate adjustments to the model's parameters, which helps minimize the loss function and improve model accuracy.


  ```python
# Training loop
for epoch in range(4):
        camembert_model.train()
        for input_ids, attention_mask, labels in train_dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = camembert_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            model_optimizer.step()
            model_optimizer.zero_grad()

        # Evaluation after each epoch
        val_loss, val_preds, val_labels = evaluate_model_performance(camembert_model, val_dataloader)
        print(f"Epoch {epoch} completed. Validation Loss: {val_loss}")
 ```

5. **Evaluation:**

   - The model's performance was evaluated using detailed classification reports, providing insights into its accuracy and effectiveness.

### Results

The model achieved the following validation loss and accuracy across epochs:

#### Phase 1:
- **Epoch 0:** Validation Loss: 1.069305415948232
- **Epoch 1:** Validation Loss: 1.0766801476478576
- **Epoch 2:** Validation Loss: 1.038533127307892
- **Epoch 3:** Validation Loss: 1.1206878264745077

**Classification Report:**

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| A1     | 0.8421    | 0.7033 | 0.7665   | 91      |
| A2     | 0.4800    | 0.6667 | 0.5581   | 72      |
| B1     | 0.6180    | 0.5978 | 0.6077   | 92      |
| B2     | 0.4725    | 0.6143 | 0.5342   | 70      |
| C1     | 0.4906    | 0.3662 | 0.4194   | 71      |
| C2     | 0.7183    | 0.6071 | 0.6581   | 84      |
| **Accuracy**      |           |        | 0.5979   | 480     |
| **Macro Avg**     | 0.6036    | 0.5926 | 0.5907   | 480     |
| **Weighted Avg**  | 0.6173    | 0.5979 | 0.6006   | 480     |

**Overall Accuracy: 0.5979**

6. **Retraining on the full dataset:**
   
- After the initial training, the model was fine-tuned on the combined full dataset to enhance its performance before making predictions. 

**Kaggle Results:**

At beginning we got **0.605** with this model. After changing the parameters, when submitted to Kaggle 8 minutes after the deadline, our model achieved an accuracy of **0.619**. This improvement in accuracy can be attributed to several factors:
- **Better Generalization:** The model's ability to generalize better on unseen data due to robust training practices.
- **Effective Use of Class Weights:** The class weights helped in handling class imbalance effectively, leading to improved performance on diverse datasets.
- **Comprehensive Fine-Tuning:** Training on the full dataset after initial evaluation enhanced the model's capability to understand the nuances of the French language, contributing to better predictions.

These results indicate that the CamemBERT model outperforms simpler models in predicting French text difficulty. The classification report highlights the precision, recall, and F1-scores across different difficulty levels, showcasing the model's effectiveness in handling varied linguistic complexities.

You can find the csv submissions here:
- For **0.619**: [`Data/new_attempt.csv`](Data/new_attempt.csv)
- For **0.605**: [`Data/submission_camembert_trying8.csv`](Data/submission_camembert_trying8.csv)

### Conclusion

CamemBERT has demonstrated its capability in predicting the difficulty of French texts, providing robust performance and detailed insights into text classification tasks. The use of class weights and a custom dataset class contributed significantly to the model's accuracy.

The saved model and tokenizer facilitate easy deployment, making CamemBERT a practical tool for applications requiring French text difficulty assessment. This project underscores the potential of advanced NLP models in specialized language tasks, achieving superior results compared to simpler approaches.

All the necessary codes can be found in the following notebook:
- [`Notebooks/CamemBERT.ipynb`](Notebooks/CamemBERT.ipynb)

## FlauBert Model 👨🏻‍🦳

### The Model

- **Training Data**: FlauBERT was trained on a variety of French texts, including books, newspapers, and web content, covering a wide range of topics and writing styles. This extensive training helps the model capture the diversity of the French language.
- **Architecture**: Similar to BERT, FlauBERT uses a transformer-based architecture, allowing it to process text in a bidirectional manner. This means it can consider the context from both directions, providing a more nuanced understanding of each word.
- **Applications**: FlauBERT can be used for various NLP tasks such as text classification, named entity recognition, and sentiment analysis. In our case, we leverage its capabilities to classify the CEFR level of French texts.

### Methodology

For this model, we used the same methodology as for our final model, with one difference: we implemented an early stopping mechanism to prevent overfitting instead of fixing the number of epochs to 4. However, this model yielded different results.

### Results

The model achieved the following validation loss and accuracy across epochs:

#### Phase 1:
- **Epoch 0:** Validation Loss: 1.31989533106486
- **Epoch 1:** Validation Loss: 1.0131553808848064
- **Epoch 2:** Validation Loss: 1.0798547307650248
- **Epoch 3:** Validation Loss: 1.0159024477005005
- **Epoch 4:** Validation Loss: 1.1687284509340923
- **Early stopping triggered.**

**Classification Report:**

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| A1     | 0.7379    | 0.8352 | 0.7835   | 91      |
| A2     | 0.6531    | 0.4444 | 0.5289   | 72      |
| B1     | 0.5537    | 0.7283 | 0.6291   | 92      |
| B2     | 0.4744    | 0.5286 | 0.5000   | 70      |
| C1     | 0.5538    | 0.5070 | 0.5294   | 71      |
| C2     | 0.7031    | 0.5357 | 0.6081   | 84      |
| **Accuracy**      |           |        | 0.6104   | 480     |
| **Macro Avg**     | 0.6127    | 0.5965 | 0.5965   | 480     |
| **Weighted Avg**  | 0.6181    | 0.6104 | 0.6061   | 480     |

**Overall Accuracy: 0.6104**

**Kaggle Results:**

When submitted to Kaggle, our model achieved an accuracy of **0.585**. The lower accuracy can be attributed to several factors:
- **Less Effective Generalization:** The model's ability to generalize on unseen data may not be as robust due to less comprehensive training practices.
- **Ineffective Use of Class Weights:** The handling of class imbalance might not have been as effective, leading to poorer performance on diverse datasets.
- **Insufficient Fine-Tuning:** The training on the full dataset after initial evaluation may not have been as thorough, affecting the model's capability to grasp the nuances of the French language.

These results indicate that the FlauBert model is not as efficient as the CamemBert model above in predicting French text difficulty. The classification report highlights the precision, recall, and F1-scores across different difficulty levels, showcasing the CamemBERT model's superior effectiveness in handling varied linguistic complexities.

You can find the csv submission and the notebook here:
- [`Data/flaubert_difficulty_predictions.csv`](Data/flaubert_difficulty_predictions.csv)
- [`Notebooks/FlauBERT.ipynb`](Notebooks/FlauBERT.ipynb)


## Data Augmentation 📈

Building upon our final model that predicts the difficulty level of French texts using CamemBERT, we explored the implementation of data augmentation techniques to further enhance the model's performance.

### Motivation

Data augmentation involves generating additional training data by modifying existing data. This technique can help improve the robustness and generalization of machine learning models, especially when dealing with limited datasets. For our project, we aimed to increase the diversity of the training data and address class imbalances by augmenting the dataset with synthetic examples.

### Methods of Data Augmentation

Several data augmentation techniques were considered and implemented:

1. **OpenAI API:**
Using OpenAI's language models to generate paraphrased versions of the original sentences, providing diverse training examples. [`Notebooks/OpenAI.ipynb`](Notebooks/OpenAI.ipynb)

 ```python
import openai
import pandas as pd

# Initialiser l'API OpenAI
openai.api_key = 'sk-proj-...'

# Charger le dataset existant
df = pd.read_csv('https://raw.githubusercontent.com/alexandrastna/French-text-using-AI/main/training_data.csv')

# Fonction pour générer des phrases basées sur le niveau de difficulté
def generate_sentence(level, examples):
    # Créez un prompt en utilisant des exemples du dataset
    prompt = f"Generate a sentence at the {level} level in French. Here are some examples:\n"
    for example in examples:
        prompt += f"- {example}\n"
    prompt += "New sentence:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # ou "gpt-4" selon votre accès
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates sentences based on the given level."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()
 ```
   
2. **Back-Translation:**
Translating the text to another language and then back to French to create paraphrased versions, introducing variability in sentence structure and vocabulary.
   
3. **Synonym Replacement:**
Randomly replacing words in the text with their synonyms to create new training examples, enhancing lexical diversity. [`Notebooks/Synonym_Remplacement.ipynb`](Notebooks/Synonym_Remplacement.ipynb)

 ```python
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv('https://raw.githubusercontent.com/alexandrastna/French-text-using-AI/main/training_data.csv')
df_unlabelled = pd.read_csv('https://raw.githubusercontent.com/alexandrastna/French-text-using-AI/main/unlabelled_test_data.csv')

# Encodage des labels de difficulté
label_encoder = LabelEncoder()
df['difficulty_encoded'] = label_encoder.fit_transform(df['difficulty'])

# Fonction pour obtenir des synonymes d'un mot
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonyms.add(synonym)
    return list(synonyms)
 ```
    
4. **Word Swapping:**
Swapping the positions of words within a sentence to generate variations, helping the model learn to handle different word orders. [`Notebooks/Word_Swapping.ipynb`](Notebooks/Word_Swapping.ipynb)

### Conclusion

We implemented various data augmentation techniques, including using the OpenAI API to generate paraphrased sentences, back-translation, synonym replacement, and word swapping, on our best models. However, these techniques did not lead to improved performance. Potential explanations for this outcome include: the complexity of the language model already capturing sufficient variability, the introduction of noise and irrelevant variations through augmentation, or the possibility that the augmented data did not adequately represent the actual distribution of the data. Consequently, the model might have struggled to generalize from the augmented examples to real-world data.

## User Interface 💻

### Objective

This interactive application aims to recommend YouTube videos based on user preferences and French language difficulty. The goal is to help users improve their French language skills by suggesting relevant videos based on their interests and language proficiency levels.

#### Specific Objectives

- Build an interactive web page that gathers user requirements and returns recommended content.
- Communicate with the YouTube API to retrieve and analyze video data.

### Coding

#### Integration of our Final Model 🥇

In the main code of our user interface, we integrated our pre-trained prediction model, that we saved in the python notebook where we created our model, to work with the user interface. The model file, `camembert_model.pth`, is located at `/Users/theo/Desktop/Data science and machine learning/Competition/`. Since the model is too big to be downloaded on GitHub, we ran the model on a Mac terminal, using these codes:

```python
pip3 install streamlit google-api-python-client youtube-transcript-api transformers torch sentencepiece tqdm pandas
streamlit run UI_YouTube.py
```

This integration allows the application to evaluate the difficulty of French YouTube video subtitles and provide recommendations accordingly.

#### Data Extraction via API
To enable communication with the YouTube API, we created an API key using the YouTube Data API v3 on Google Cloud. This key is necessary to authenticate and make authorized requests to the YouTube API.

The application uses the YouTube API to search for videos based on given keywords. The `get_video_list(search_keyword)` function retrieves videos based on user-provided keywords. Titles, descriptions, and thumbnail URLs of the videos are stored in a DataFrame for further processing and analysis. If French subtitles are not available, English subtitles are used as a fallback.

```python
def get_video_list(search_keyword):
    search_response = perform_search(youtube_service, q=search_keyword, maxResults=50, relevanceLanguage="fr", videoCaption="closedCaption")
    video_items = search_response.get("items", [])
    if not video_items:
        st.write("No videos found for this keyword.")
        return pd.DataFrame()
    video_dataframe = pd.DataFrame(columns=['video_url', 'title', 'description', 'thumbnail_url', 'transcript'])
    for video_item in tqdm.tqdm(video_items):
        video_id = video_item["id"]["videoId"]
        video_response = fetch_video_details(youtube_service, id=video_id)
        title, description, thumbnail_url = parse_video_info(video_response)

        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['fr'])
        except:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            except:
                continue

        transcript_text = '. '.join([entry['text'] for entry in transcript])
        transcript_text = transcript_text.replace('\n', ' ')
        video_data = {'video_url': 'https://www.youtube.com/watch?v=' + video_id, 'title': title, 'description': description, 'thumbnail_url': thumbnail_url, 'transcript': transcript_text}
        video_dataframe = pd.concat([video_dataframe, pd.DataFrame([video_data])], ignore_index=True)

    return video_dataframe
```
#### Predict and Recommend with our Final Model 🥇
The application requires the user to input two parameters: a keyword and a difficulty level. After obtaining this information, the application searches for French-tagged videos on YouTube, retrieves subtitles, and uses our final model to predict the difficulty of the videos. Videos matching the user's difficulty level are then recommended.

```python
def video_recommender(keyword, proficiency_level):
    video_data = get_video_list(keyword)
    if video_data.empty:
        return pd.DataFrame()

    feature_vectors = extract_features(video_data['transcript'], max_length=512)
    if feature_vectors.size == 0:
        return pd.DataFrame()

    difficulty_predictions = feature_vectors.argmax(axis=1)
    video_data['difficulty'] = pd.Series(difficulty_predictions).map({0:'A1', 1:'A2', 2:'B1', 3:'B2', 4:'C1', 5:'C2'})
    filtered_videos = video_data[video_data['difficulty'] == proficiency_level].reset_index()
    return filtered_videos
```

#### Interactive User Interface
The Streamlit user interface allows users to input a keyword and select a difficulty level. The recommended videos are displayed with their title, URL, and thumbnail, allowing users to preview and access the content directly.

![User Interface](Images/UI.png)

Users can input a keyword in the "Mot-clé" field and select their desired French language proficiency level from the "Niveau" dropdown menu. After clicking the "Afficher les vidéos recommandées" button, the application fetches and displays videos that match the criteria. In this example, the keyword "Banane" was used, and the difficulty level selected was C2. The application successfully recommended 47 videos, providing their titles, URLs, and thumbnails for easy access.

### Results and Limitations

#### Results
The application classifies YouTube videos based on their linguistic difficulty and user preferences. It provides an intuitive user interface to display recommended videos using the model we created that predict the difficulty of a given text.

#### Limitations
- Availability of Subtitles: If French subtitles are not available, English subtitles are used, which may affect classification accuracy.
- YouTube API Quota: The application is limited by the YouTube API quota.
- Performance: Processing videos can take time, especially for long videos or when processing many videos. Retrieving subtitles, extracting features, and classifying may take several seconds.
- Subtitle Length: The CamemBert model is limited to 512 tokens, which can result in information loss for longer subtitles.

All the necessary codes can be found in the following notebook:
- [`Streamlit/UI_YouTube.py`](Streamlit/UI_YouTube.py)

## Our Video 📹

Please find our video here: https://m.youtube.com/watch?v=ZeRtaXTgLRs&feature=youtu.be&cbrd=1&cbrd=1
