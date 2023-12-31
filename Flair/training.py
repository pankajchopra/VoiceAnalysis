from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import os
# Path to your labeled sentiment dataset (CSV file)
train_file = 'labeled_sentiment_dataset.csv'
data_folder = os.path(train_file)

# Specify the file or folder name
file_name = "labeled_sentiment_dataset.csv"

# Construct the full path
full_path = os.path.join(".\\", file_name)

# Load the dataset
corpus: Corpus = CSVClassificationCorpus(data_folder, column_name_map={'text': 0, 'label': 1}, skip_header=True)

# Define word embeddings (you can customize this based on your needs)
word_embeddings = [WordEmbeddings('glove')]

# Define document embeddings
document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)

# Define the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)

# Define the trainer
trainer = ModelTrainer(classifier, corpus)

# Train the model
trainer.train(
    'path_to_store_model',  # where to save the trained model
    learning_rate=0.1,
    mini_batch_size=32,
    max_epochs=10
)
