import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from Programs.vectorization_and_modeling import metrices
from Programs import utils
from sklearn.metrics import accuracy_score, classification_report

NLP_HOME = r'E:\Newgen\NLP\Programs'
create_data_folder = os.path.join(NLP_HOME, "Text_processing")
csv_processed_path = os.path.join(create_data_folder, "processed_dataset.csv")
data = pd.read_csv(csv_processed_path)
X_train, X_test, y_train, y_test = train_test_split(data['Processed_text'], data['Class'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Use Logistic Regression instead of SVC
classifier = LogisticRegression()
classifier.fit(X_train_bow, y_train)

y_pred = classifier.predict(X_test_bow)
print("y_pred", y_pred[0])

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

classification_rep = classification_report(y_test, y_pred, output_dict=True)
precision = classification_rep['weighted avg']['precision']
recall = classification_rep['weighted avg']['recall']

precision_recall_path = r'E:\Proj\Programs\vectorization_and_modeling\logistic_regression\BOW\accuracy_precision_recall.txt'
utils.create_text_file(precision_recall_path, f"Accuracy: {accuracy:.2f} \nPrecision: {precision:.2f}\nRecall: {recall:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

save_path = r'E:\Proj\Programs\vectorization_and_modeling\Logistic_regression\BOW\LogisticRegression'
metrices.save_classification_metrics_image(y_test, y_pred, save_path)
