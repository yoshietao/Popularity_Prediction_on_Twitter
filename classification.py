###########################
# Author: Te-Yuan Liu
###########################

###########################
# Import Packages
###########################
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
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
import re
from random import randint
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
        with open('../tweet_data/'+filename+'.txt') as data:
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
                    y_list.append(0)
                elif location.find('Massachusetts') is not -1 or location.find('MA') is not -1 or location.find('Boston') is not -1:
                    X_list.append(text)
                    y_list.append(1)
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
    
def svd_selection(X_train, y_train, X_test, y_test, class_names):
    index_list = [0]
    X_train_r, X_test_r = X_train[:, np.array(index_list)], X_test[:, np.array(index_list)]
    acc_best = log_analysis(X_train_r, y_train, X_test_r, y_test, class_names)
    for i in range(1, 50, 1):
        print('##########################')
        print('component of index: ', i, 'is added...')
        index_list.append(i)
        X_train_r, X_test_r = X_train[:, np.array(index_list)], X_test[:, np.array(index_list)]
        acc_cur = log_analysis(X_train_r, y_train, X_test_r, y_test, class_names)
        print('currnet acc: ', acc_cur)
        if acc_cur > acc_best:
            acc_best = acc_cur
        else:
            print('component removed...')
            index_list.pop()
    index_list_pop = index_list[1:]
    X_train_r, X_test_r = X_train[:, np.array(index_list_pop)], X_test[:, np.array(index_list_pop)]
    acc_cur = log_analysis(X_train_r, y_train, X_test_r, y_test, class_names)
    if acc_cur > acc_best:
        print('index_list: ', index_list_pop)
        print('best_acc: ', acc_cur)
    else:
        print('index_list', index_list)
        print('best acc: ', acc_best)

def svm_analysis(X_train, y_train, X_test, y_test, class_names):
    print('###########################')
    print('Support Vector Machine: ')
    print('###########################')
    svm_clf = svm.SVC(C=1000, probability=True)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    y_pred_proba = svm_clf.predict_proba(X_test)
    report_results(y_test, y_pred, y_pred_proba, class_names)

def log_analysis(X_train, y_train, X_test, y_test, class_names, optimize=False):
    print('###########################')
    print('Logistic Regression: ')
    print('###########################')
    acc_best, c_best = -1, 100
    if optimize:
        for x in range(-3, 4):
            c = pow(10, x)
            log_clf = LogisticRegression(C=c, random_state=42)
            acc = cross_val_score(log_clf, X_train, y_train, cv=10, scoring='accuracy')
            if acc.mean() > acc_best:
                acc_best = acc.mean()
                c_best = c
        print('Best C: ', c_best)
        print('Best validation acc: ', acc_best)
    
    log_clf = LogisticRegression(C=c_best, random_state=42)
    log_clf.fit(X_train, y_train)
    y_pred = log_clf.predict(X_test)
    y_pred_proba = log_clf.predict_proba(X_test)
    report_results(y_test, y_pred, y_pred_proba, class_names)

def rf_analysis(X_train, y_train, X_test, y_test, class_names, optimize=False):
    print('###########################')
    print('Random Forest: ')
    print('###########################')
    acc_best, depth_best = -1, 10
    if optimize:
        for d in range(5, 51, 10):
            rf_clf = RandomForestClassifier(max_depth=d, random_state=42)
            acc = cross_val_score(rf_clf, X_train, y_train, cv=10, scoring='accuracy')
            if acc.mean() > acc_best:
                acc_best = acc.mean()
                depth_best = d
        print('Best max_depth: ', depth_best)
        print('Best validation acc: ', acc_best)

    rf_clf = RandomForestClassifier(max_depth=depth_best, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    y_pred_proba = rf_clf.predict_proba(X_test)
    report_results(y_test, y_pred, y_pred_proba, class_names)

def mlp_analysis(X_train, y_train, X_test, y_test, class_names):
    print('###########################')
    print('Neural Network: ')
    print('###########################')
    mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 64, 32), random_state=42)
    mlp_clf.fit(X_train, y_train)
    y_pred = mlp_clf.predict(X_test)
    y_pred_proba = mlp_clf.predict_proba(X_test)
    report_results(y_test, y_pred, y_pred_proba, class_names)

def preprocess(X):
    for i in range(len(X)):
        regex = re.compile('[%s]' % re.escape(punctuation))
        X[i] = regex.sub(' ',X[i]).lower()
    return X

def noise_reduce(X, y):
    del_list = []
    for i in range(len(X)):
        if X[i].find('hawk') is -1 and X[i].find('patriot') is -1:
            del_list.append(i)
    X = np.delete(X, del_list, 0)
    y = np.delete(y, del_list, 0)
    return X, y

def random_reduce(X, y):
    del_list = list(np.random.choice(len(X), 27990, replace=False))
    X = np.delete(X, del_list, 0)
    y = np.delete(y, del_list, 0)
    return X, y

def baseline(X, y, class_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    sum_ = 0.
    y_pred = []
    central = 0.
    h_m = 0.
    h_w = 0.
    p_m = 0.
    p_w = 0.
    counter = 0
    w = 0.
    a,b,c,d=0,0,0,0
    
    for i in range(len(X)):
        counter += 1
        if y[i] == 0:
            w += 1.
        if (X[i].find('hawk') is -1 and X[i].find('patriot') is not -1) and y[i] == 1:
             p_m += 1. 
        elif (X[i].find('hawk') is -1 and X[i].find('patriot') is not -1) and y[i] == 0:
            p_w += 1.
        elif (X[i].find('hawk') is not -1 and X[i].find('patriot') is -1) and y[i] == 1:
            h_m += 1.
        elif (X[i].find('hawk') is not -1 and X[i].find('patriot') is -1) and y[i] == 0:
            h_w += 1.
        elif (X[i].find('hawk') is -1 and X[i].find('patriot') is -1) and y[i] == 1:
            a += 1.
            #print(X[i])
        elif (X[i].find('hawk') is -1 and X[i].find('patriot') is -1) and y[i] == 0:
            b += 1.
            #print(X[i])
        elif (X[i].find('hawk') is not -1 and X[i].find('patriot') is not -1) and y[i] == 1:
            c += 1.
        elif (X[i].find('hawk') is not -1 and X[i].find('patriot') is not -1) and y[i] == 0:
            d += 1.
    print('WA portion: ', w/len(X))
    print('Patriots from MA: ', p_m/len(X))
    print('Patriots from WA: ', p_w/len(X))
    print('Gohawks from MA: ', h_m/len(X))
    print('Gohawks from WA: ', h_w/len(X))
    print('None from MA: ', a/len(X))
    print('None from WA: ', b/len(X))
    print('Both from MA: ', c/len(X))
    print('Both from WA: ', d/len(X))
    print((p_m+p_w+h_m+h_w+a+b+c+d)/len(X))
    
    for i in range(len(X_test)):
        s = randint(0, 1)
       
        if X_test[i].find('hawk') is not -1:
            s = 0
        if X_test[i].find('patriot') is not -1:
            s = 1
        if y_test[i] == s:
            sum_ += 1.
        y_pred.append(s)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('Recall: ', metrics.recall_score(y_test, y_pred))
    print('Precisoin: ', metrics.precision_score(y_test, y_pred))
    # plot confusion matrix
    print('Plot Confusion Matrix')
    cm = metrics.confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names=class_names)

###########################
# Main
###########################
def main():
    print('starting part 2...')
    class_names = ['Washington', 'Massachusetts']
    X, y = generate_X_y('tweets_#superbowl', True)
    #X = preprocess(X)
    #X, y = noise_reduce(X, y)
    #baseline(X,y, class_names)
    
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

    #svd_selection(X_train_tf_svd, y_train, X_test_tf_svd, y_test, class_names)
    svm_analysis(X_train_tf_svd, y_train, X_test_tf_svd, y_test, class_names)
    log_analysis(X_train_tf_svd, y_train, X_test_tf_svd, y_test, class_names, True)
    rf_analysis(X_train_tf_svd, y_train, X_test_tf_svd, y_test, class_names, True)
    mlp_analysis(X_train_tf_svd, y_train, X_test_tf_svd, y_test, class_names)  
    
if __name__ == "__main__":
    main()




