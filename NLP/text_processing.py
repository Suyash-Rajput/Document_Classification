import string
import nltk
import os
import emoji
import pandas as pd
from Programs import utils
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('omw-1.4')

def text_processing(text):
    encoded_text = emoji.demojize(text)      # convert emoji to text 
    encoded_text = ''.join([char for char in encoded_text if char.isalnum() or char.isspace()])
    sentences = sent_tokenize(encoded_text)      # Tokenization    
    words = [word_tokenize(sentence) for sentence in sentences]
    stop_words = set(stopwords.words('english'))       # Stop words removal
    filtered_words = [[word for word in sentence if word.lower() not in stop_words and word.isalnum()] for sentence in words]
    stemmer = PorterStemmer()      # Stemming
    stemmed_words = [[stemmer.stem(word) for word in sentence] for sentence in filtered_words]
    lemmatizer = WordNetLemmatizer()     # Lemmatization
    lemmatized_words = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in stemmed_words]
    complete_string = ', '.join("'" + item + "'" for item in lemmatized_words[0])
    return complete_string

def create_processed_text(NLP_HOME):
    create_data_folder = os.path.join(NLP_HOME, "Text_processing") 
    Dataset_Generator = os.path.join(NLP_HOME, "Dataset_Generator") 
    csv_file_path = os.path.join(Dataset_Generator, "dataset.csv")
    csv_processed_path =  os.path.join(create_data_folder, "processed_dataset.csv")
    file_encoding = "latin1"
    df = pd.read_csv(csv_file_path, encoding=file_encoding)
    df['Processed_text'] = df['Text'].apply(text_processing)
    del df['Text']
    df.to_csv(csv_processed_path, index=False)
    print(f"Successfully Processed dataset is saved to {csv_processed_path}")
    
if __name__ == "__main__":
    NLP_HOME = r'E:\Newgen\NLP\Programs'
    create_processed_text(NLP_HOME)    