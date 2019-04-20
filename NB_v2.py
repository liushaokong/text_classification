#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this is the solution for the NB classification, it shows how to import data and create labels,
and use the MultinomialNB to predict the test label, after vectorizing the doc.
Hyper perameter alpha in NB is varied to get the best result.
"""

# 1. import necessary modules
import os
import io
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# from sklearn.model_selection import train_test_split


# 2. data preparation
train_dir = './text classification/train'
test_dir = './text classification/test'

stop_words = [line.split('\n')[0] for line in
              io.open('./text classification/stop/stopword.txt', encoding='UTF-8').readlines()]
# print(len(stop_words))


def get_file(file_dir):
    """
    to get all the file and create the label
    :param file_dir:
    :return: read all documents into 1 list, and return the corresponding labels
    """
    docs = []
    # doc_names = []
    labels = []

    for files in os.listdir(file_dir):
        for file in os.listdir(file_dir + '/' + str(files)):
            labels.append(os.listdir(file_dir).index(files))

            file_name = file_dir + '/' + str(files) + '/' + str(file)
            # doc_names.append(file_name)
            docs.append(io.open(file_name, encoding='GB18030').readlines())

    return docs, labels


def get_word_list(doc):
    word_list = []

    for sentence in doc:
        temp_list = jieba.lcut(sentence)
        word_list.append(temp_list)

    return word_list


def toWordString(docs):
    contents = []
    for doc in docs:
        content = ''
        for word in doc:
            if word != '，' or word != '，':
                content += word + ' '
        contents += [content]
    return contents


train_docs, train_labels = get_file(train_dir)
test_docs, test_labels = get_file(test_dir)

train_words = [get_word_list(doc)[0] for doc in train_docs]
test_words = [get_word_list(doc)[0] for doc in test_docs]

train_contents = toWordString(train_words)
test_contents = toWordString(test_words)


# 3. 计算单词的权重
Vec = TfidfVectorizer(stop_words=stop_words)  # max_df=1
train_features = Vec.fit_transform(train_contents)
# print(len(train_features.toarray()[10]))


# 6. changing alpha
alphas = [0.001, 0.01, 0.1, 1]  # 1 for Laplace, <1 for Lidstone
for alpha in alphas:
    # 4. clf train
    clf = MultinomialNB(alpha=alpha)
    clf.fit(train_features, train_labels)

    # 5. acc_score
    test_features = Vec.transform(test_contents)
    predict_labels = clf.predict(test_features)
    acc_score = metrics.accuracy_score(test_labels, predict_labels)
    print('alpha=%.3lf,' % alpha, 'acc_score=%.3lf' % acc_score)

