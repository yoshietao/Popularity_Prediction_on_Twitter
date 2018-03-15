###########################
# Author: Te-Yuan Liu
###########################

###########################
# Import Packages
###########################
import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from string import punctuation
import collections
import itertools
import os
###########################
# Define Functions
###########################
def generate_X_y(filename, recreate=False):
    if Path('./data/X.npy').exists() and Path('./data/y.npy').exists() and not recreate:
        print('data found...')
        return np.load('./data/X.npy'), np.load('./data/y.npy')
    else:
        if not Path('./data').exists():
            os.makedirs('./data')
        print('create data...')
        X_list, y_list = [], []
        counter = 0
        with open(filename+'.txt') as data:
            for line in data:
                line = json.loads(line)
                text = line['title']
                location = line['tweet']['user']['location']
                #print(counter)
                #if counter > 1000:
                #    break
                #if counter > 100:
                    #print(text)
                    #print(location)
                if location.find('Washington') is not -1 or location.find('WA') is not -1 or location.find('Seattle') is not -1:
                    X_list.append(text)
                    y_list.append(0.)
                elif location.find('Massachusetts') is not -1 or location.find('MA') is not -1 or location.find('Boston') is not -1:
                    X_list.append(text)
                    y_list.append(1.)
                    counter += 1
        print('size of WA data: ', len(y_list) - counter)
        print('size of MA data: ', counter)
        np.save('./data/X.npy', np.array(X_list))
        np.save('./data/y.npy', np.array(y_list))
        return X_list, y_list

def generate_vectorizer(min_df, max_df=1.0):
    stop_words_skt = text.ENGLISH_STOP_WORDS
    stop_words_en = stopwords.words('english')
    combined_stopwords = set.union(set(stop_words_en), set(punctuation), set(stop_words_skt))
    stemmer = SnowballStemmer('english')
    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            analyzer = super(StemmedCountVectorizer, self).build_analyzer()
            return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    vectorizer = StemmedCountVectorizer(min_df=min_df, analyzer='word', stop_words = combined_stopwords)
    return vectorizer

def plot_confusion_matrix(cm, class_names, cmap=plt.cm.Blues):
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=30)
    plt.yticks(tick_marks, class_names)
    fmt = 'd'
    thresh = (cm.min() + cm.max())/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    plt.show()

def plot_roc(fpr, tpr):
    fig, ax = plt.subplots()
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label='area under curve = %0.4f' % roc_auc)
    ax.grid(color='0.7', linestyle='--', linewidth=1)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel('False Positive Rate', fontsize=15)
    ax.set_ylabel('True Positive Rate', fontsize=15)
    ax.legend(loc='lower right')
    ax.set_title('ROC Curve')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(15)
    plt.show()

def report_results(y_true, y_pred, y_pred_proba, class_names):
    print('Accuracy: ', metrics.accuracy_score(y_true, y_pred))
    print('Recall: ', metrics.recall_score(y_true, y_pred))
    print('Precisoin: ', metrics.precision_score(y_true, y_pred))
    # plot ROC curve
    print('Plot ROC Curve')
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:,1])
    plot_roc(fpr, tpr)
    # plot confusion matrix
    print('Plot Confusion Matrix')
    cm = metrics.confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names=class_names)

def svm_analysis(X_train, y_train, X_test, y_test, class_names):
    print('###########################')
    print('Support Vector Machine: ')
    print('###########################')
    svm_clf = svm.SVC(C=1000, probability=True)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    y_pred_proba = svm_clf.predict_proba(X_test)
    report_results(y_test, y_pred, y_pred_proba, class_names)

def log_analysis(X_train, y_train, X_test, y_test, class_names):
    print('###########################')
    print('Logistic Regression: ')
    print('###########################')
    log_clf = LogisticRegression(C=1000, random_state=42)
    log_clf.fit(X_train, y_train)
    y_pred = log_clf.predict(X_test)
    y_pred_proba = log_clf.predict_proba(X_test)
    report_results(y_test, y_pred, y_pred_proba, class_names)

def rf_analysis(X_train, y_train, X_test, y_test, class_names):
    print('###########################')
    print('Random Forest: ')
    print('###########################')
    rf_clf = RandomForestClassifier(max_depth=20, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    y_pred_proba = rf_clf.predict_proba(X_test)
    report_results(y_test, y_pred, y_pred_proba, class_names)

def mlp_analysis(X_train, y_train, X_test, y_test, class_names):
    print('###########################')
    print('Neural Network: ')
    print('###########################')
    mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 64), random_state=42)
    mlp_clf.fit(X_train, y_train)
    y_pred = mlp_clf.predict(X_test)
    y_pred_proba = mlp_clf.predict_proba(X_test)
    report_results(y_test, y_pred, y_pred_proba, class_names)

###########################
# Main
###########################
def main():
    print('starting part 2...')
    X, y = generate_X_y('tweets_#superbowl', False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    # create TFxIDF vector representations
    vectorizer = generate_vectorizer(min_df=5)
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)

    tf_transformer = TfidfTransformer()

    X_train_tf = tf_transformer.fit_transform(X_train_counts)
    X_test_tf = tf_transformer.transform(X_test_counts)
    
    svd = TruncatedSVD(n_components=50, algorithm='randomized', n_iter=10, random_state=42)
    X_train_tf_svd = svd.fit_transform(X_train_tf)
    X_test_tf_svd = svd.transform(X_test_tf)
    '''
    nmf = NMF(n_components=200, init='random', random_state=42)
    X_train_tf_nmf = nmf.fit_transform(X_train_tf)
    X_test_tf_nmf = nmf.transform(X_test_tf)
    '''
    class_names = ['Washington', 'Massachusetts']
    svm_analysis(X_train_tf_svd, y_train, X_test_tf_svd, y_test, class_names)
    log_analysis(X_train_tf_svd, y_train, X_test_tf_svd, y_test, class_names)
    rf_analysis(X_train_tf_svd, y_train, X_test_tf_svd, y_test, class_names)
    mlp_analysis(X_train_tf_svd, y_train, X_test_tf_svd, y_test, class_names)
    
if __name__ == "__main__":
    main()




