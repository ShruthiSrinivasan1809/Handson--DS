# Handson--DS
This project is developed to perform offensive text classification by using Python on Google Colab. Random forest and Decision tree-based classifiers are used for this project. This experiment is carried out using 
3 different datasets for train, validation and test. The offensive words in the train set are categorized as OFF while the normal words are categorized as  NOT OFF. The necessary packages are imported To preprocess the dataset,
we utilized stemming techniques to remove inflections from the words, which helped to reduce the complexity of the dataset. We also eliminated stop words, which are words that do not carry significant meaning in
the context of the tweets. This helped to streamline the dataset and focus on the most important words. Additionally, we removed irrelevant characters that were not relevant to the tone of the tweet. 

Once the dataset was preprocessed, we employed the tfidf vectorization technique to convert the tweets into scores. This technique takes into account the frequency of words in each tweet, as well
as their overall frequency in the entire dataset. By using tfidf vectorization, we were able to capture the important words within each tweet and assign them scores based on their relevance to the overall
dataset. We only applied this vectorization process to the training dataset, and the resulting vectorizer model was stored. Next, we utilized the resulting vectorizer model
and applied it to the training data, and the resulting values and labels were used to train our machine learning model. We calculated the accuracy of the training dataset using this process. Similarly,
we applied the validation dataset directly to the vectorizer model and used the resulting tweets and labels to make predictions on the machine learning model. We obtained the accuracy scores for both the training and validation datasets using the entire
original dataset.

We also split the training dataset into four different subsets, representing 25%, 50%, 75%, and 100% of the original dataset. We repeated the above process for each of these subsets, obtaining validation
results for each corresponding sample, and storing the vectorizer and model for each dataset. Using the vectorizer and model for all four splits, we applied test input to each of the models to obtain
corresponding accuracy scores. By using this approach, we were able to evaluate the accuracy of our machine-learning models across different dataset sizes and assess their performance on new,
unseen data.
