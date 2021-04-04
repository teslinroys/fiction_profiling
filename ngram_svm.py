#   Ngram linear SVM classifier for time periods
#   Teslin Roys

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import argsort
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.svm import LinearSVC

def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 10))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=75, ha='right')
    plt.show()

DATA_DIR = 'C:/Users/Administrator/Documents/auth_att/cent_chunked'
data = load_files(DATA_DIR, encoding="utf-8", decode_error="replace")
labels, counts = np.unique(data.target, return_counts=True)
labels_str = np.array(data.target_names)[labels]
print(dict(zip(labels_str, counts)))

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.30, random_state=1234)

vectorizer = TfidfVectorizer(ngram_range=(1,2), analyzer='word')

vectorizer.fit(X_train)
vectorizer.fit(X_test)
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=2)
#clf.fit(X_train_vec, y_train)

svm = LinearSVC()
svm.fit(X_train_vec, y_train)

plot_coefficients(svm, vectorizer.get_feature_names())

y_pred = svm.predict(vectorizer.transform(X_test))

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

# y_pred = clf.predict(vectorizer.transform(X_test))
# print(accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# TODO: Display feature_importances more prettily. Methodology: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html