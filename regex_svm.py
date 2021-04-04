import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from numpy import argsort
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import nltk
import re
nltk.download('punkt')

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

def get_occurences(t,rgx):
    occ = len(rgx.findall(t.lower()))
    return occ

def get_sentence_length(t):
    sent_text = nltk.sent_tokenize(t)
    total_len = 0
    for s in sent_text:
        total_len = total_len + len(s)
    r = total_len / len(sent_text)
    return r

def get_text_length(x):
    return np.array([get_sentence_length(t) for t in x]).reshape(-1, 1)

contr_re = re.compile("n't")
def get_contractions(x):
    return np.array([get_occurences(t, contr_re) for t in x]).reshape(-1, 1)

br_re = re.compile("(king|queen|squire|castle|sir|ma'am|hither|england|london|scrooge|pounds|shall|lord|grey)")
am_re = re.compile("(dollar|new\syork|gray|america|city|parlor|boston|washington)")
def get_content_words(x):
    return np.array([get_occurences(t, am_re) + get_occurences(t, br_re) for t in x]).reshape(-1, 1)

#post-paper note: ze performs muuuuuch better than the original (i|y)z(e|i|a) form
ize_re = re.compile(r"(\w*(i|y)ze\b)")
def get_ize(x):
    return np.array([get_occurences(t, ize_re) for t in x]).reshape(-1, 1)

spec_re = re.compile("(ç|é|â|ê|î|ô|û|à|è|ì|ò|ù|ë|ï|ü)")
def get_special_char(x):
    return np.array([get_occurences(t, spec_re)  for t in x]).reshape(-1, 1)

nor_re = re.compile(r"\bnor\b")
def get_nor(x):
    return np.array([get_occurences(t, nor_re)  for t in x]).reshape(-1, 1)

prog_re = re.compile(r"((\s*(was|were|am|is|are|'s|'re|'m)\sbeing\s[a-zA-Z](ed|ied|d)\s)|\s(was|were|am|is|are|'s|'re|'m)\sbeing\s(said|made|gone|taken|come|seen|known|got|gotten|given|found|thought|told|become|shown|left|felt|put|brought|begun|kept|held|written|stood|heard|let|meant|set|met|run|paid|sat|spoken|lain|led|read|grown|lost|fallen|sent|built|understood|drawn|broken|spent|cut|risen|driven|bought|worn|chosen)\s*)")
def get_prog(x):
    return np.array([get_occurences(t, prog_re)  for t in x]).reshape(-1, 1)


classifier = Pipeline([
    ('features', FeatureUnion([
        ('ize', Pipeline([
             ('count', FunctionTransformer(get_ize, validate=False)),
         ]))
        # ('length', Pipeline([
        #     ('count', FunctionTransformer(get_text_length, validate=False)),
        # ])),
        # ('contractions', Pipeline([
        #     ('count', FunctionTransformer(get_contractions, validate=False)),
        # ])),
        # ('nor', Pipeline([
        #     ('count', FunctionTransformer(get_nor, validate=False)),
        # ])),
        # ('ize', Pipeline([
        #     ('count', FunctionTransformer(get_ize, validate=False)),
        # ])),
        # ('prog', Pipeline([
        #      ('count', FunctionTransformer(get_prog, validate=False)),
        #  ])),
    ])),
    ('clf', LinearSVC(max_iter=1250000))])

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

coefs = classifier.named_steps['clf'].coef_
print(coefs)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))