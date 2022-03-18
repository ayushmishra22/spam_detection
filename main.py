# This is the project

import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

CORRECT_DATASET_NO = [1,2,3]

Current_directory = os.getcwd()
#Path of Training set 1
#train_path_ham = ["C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset1\\train\\ham",
#                  "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset2\\enron1_train\\enron1\\train\\ham",
#                  "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset3\\enron4_train\\enron4\\train\\ham"]
#train_path_spam = ["C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset1\\train\\spam",
#                   "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset2\\enron1_train\\enron1\\train\\spam",
#                   "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset3\\enron4_train\\enron4\\train\\spam"]

train_path_ham = [Current_directory+"\\dataset1\\train\\ham",
                  Current_directory+"\\dataset2\\enron1_train\\enron1\\train\\ham",
                  Current_directory+"\\dataset3\\enron4_train\\enron4\\train\\ham"]
train_path_spam = [Current_directory+"\\dataset1\\train\\spam",
                   Current_directory+"\\dataset2\\enron1_train\\enron1\\train\\spam",
                   Current_directory+"\\dataset3\\enron4_train\\enron4\\train\\spam"]

test_path_ham = [Current_directory+"\\dataset1\\test\\ham",
                 Current_directory+"\\dataset2\\enron1_test\\enron1\\test\\ham",
                 Current_directory+"\\dataset3\\enron4_test\\enron4\\test\\ham"]
test_path_spam = [Current_directory+"\\dataset1\\test\\spam",
                  Current_directory+"\\dataset2\\enron1_test\\enron1\\test\\spam",
                  Current_directory+"\\dataset3\\enron4_test\\enron4\\test\\spam"]

#test_path_ham = ["C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset1\\test\\ham",
#                 "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset2\\enron1_test\\enron1\\test\\ham",
#                 "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset3\\enron4_test\\enron4\\test\\ham"]
#test_path_spam = ["C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset1\\test\\spam",
#                  "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset2\\enron1_test\\enron1\\test\\spam",
#                  "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset3\\enron4_test\\enron4\\test\\spam"]




def create_dataframe(bag, vectors):
    dataframe = pd.DataFrame(np.array(vectors), columns=bag)
    return dataframe

def add_class_labels(dataframe, count_class1, count_class2):
    labels = []
    for i in range(count_class1):
        labels.append(0)
    for i in range(count_class2):
        labels.append(1)
    dataframe["Class_label"] = labels
    return dataframe

def read_datasets(path_ham, path_spam):
    cur_dir = os.getcwd()
    raw_data_ham = []
    raw_data_spam = []
    os.chdir(path_ham)
    cur_dir = os.getcwd()
    print(cur_dir)
    for file in os.listdir(cur_dir):
        try:
            raw_data_ham.append(open(file, 'r', encoding='utf-8').read())
        except:
            try:
                raw_data_ham.append(open(file, 'r', encoding='latin-1').read())
            except:
                print("File can't be opened, ignoring the file")
                continue
    os.chdir(path_spam)
    cur_dir = os.getcwd()
    print(cur_dir)
    for file in os.listdir(cur_dir):
        try:
            raw_data_spam.append(open(file, 'r', encoding='utf-8').read())
        except:
            try:
                raw_data_spam.append(open(file, 'r', encoding='latin-1').read())
            except:
                print("File can't be opened, ignoring the file")
                continue
    return raw_data_ham, raw_data_spam, len(raw_data_ham), len(raw_data_spam)

def bag_of_words_train(training_data, train_count_ham, train_count_spam):
    bow_vectorizer = CountVectorizer()
    vectors = bow_vectorizer.fit_transform(training_data).toarray()
    train_dataframe = create_dataframe(bow_vectorizer.get_feature_names(), vectors)
    train_dataframe = add_class_labels(train_dataframe, train_count_ham, train_count_spam)
    return train_dataframe, bow_vectorizer

def bag_of_words_test(test_data, test_count_ham, test_count_spam, vectorizer):
    #test_vectors = vectorizer.transform(test_ham1+test_spam1).toarray()  #Creating test vectors
    test_vectors = vectorizer.transform(test_data).toarray()
    test_dataframe = create_dataframe(vectorizer.get_feature_names(), test_vectors) #Creating test dataframe
    test_dataframe = add_class_labels(test_dataframe, test_count_ham, test_count_spam)
    return test_dataframe

def training_multinomialNB(train_dataframe, class_labels):
    total_docs = len(train_dataframe)
    count_docs_in_class = {}
    prior = {}
    tct_dash = {}
    tct = {}
    cond_prob = {}
    for i in class_labels:
        count_docs_in_class[i] = len(train_dataframe[train_dataframe["Class_label"] == i])
        prior[i] = count_docs_in_class[i]/total_docs
        dataframe = train_dataframe[train_dataframe["Class_label"] == i]
        dataframe = dataframe.drop(["Class_label"],axis=1)
        tct_dash[i] = dataframe.sum(axis=1).sum(axis=0)
        tct[i] = dataframe.sum(axis=0)
        B = len(dataframe.columns.values.tolist())
        s = 0
        for t in dataframe.columns.values.tolist():
            key = str(t)+"|"+str(i)
            cond_prob[key] = (tct[i][t]+1)/(tct_dash[i]+B)
    return prior, cond_prob

def apply_multinomialNB(test_dataframe, test_vector_count, class_labels, prior, cond_prob):
    predicted = []
    for k in range(0,test_vector_count):
        score = {}
        score[k] = {}
        score[k][0] = 0
        score[k][1] = 0
        for i in class_labels:
            score[k][i] = score[k][i] + np.log(prior[i])
            for j in test_dataframe.columns[:-1]:
                if test_dataframe.at[k,j] != 0:
                    key = str(j)+"|"+str(i)
                    score[k][i] = score[k][i] + test_dataframe.at[k, str(j)]*np.log(cond_prob[key])
        predicted.append(0 if score[k][0] >= score[k][1] else 1)

    return predicted


def classifier_report(true_labels, predicted):
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted).ravel()
    precision_score = tp/(fp+tp)
    recall = tp/(fn+tp)
    accuracy = (tp+tn)/(tp+fn+fp+tn)
    f1 = (2*precision_score*recall)/(precision_score+recall)
    print("Precision="+str(precision_score))
    print("recall="+str(recall))
    print("accuracy="+str(accuracy))
    print("F1="+str(f1))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    try:
        if (int(sys.argv[1]) in CORRECT_DATASET_NO):
            dataset_no = int(sys.argv[1]) - 1
        else:
            raise ValueError("The mentioned dataset number was incorrect")
    except ValueError as err:
        print(err)
        print("The dataset number should be in the range of [1,3]")
        sys.exit(1)
    #train_ham1, train_spam1, train_count_ham1, train_count_spam1 = read_datasets(train_path_ham[0], train_path_spam[0])
    #train_dataframe1, bow_vectorizer1 = bag_of_words_train(train_ham1+train_spam1, train_count_ham1, train_count_spam1)
    #train_dataframe1 = create_dataframe(vectorizer1.get_feature_names(), vectors1)
    #train_dataframe1 = add_class_labels(train_dataframe1, train_count_ham1, train_count_spam1)
    #test_ham1, test_spam1, test_count_ham1, test_count_spam1 = read_datasets(test_path_ham[0], test_path_spam[0]) #Reading test dataset
    #test_dataframe1 = bag_of_words_test(test_ham1+test_spam1, test_count_ham1, test_count_spam1, bow_vectorizer1)
    #test_vectors = vectorizer1.transform(test_ham1+test_spam1).toarray()  #Creating test vectors
    #test_dataframe = create_dataframe(vectorizer1.get_feature_names(), test_vectors) #Creating test dataframe
    #test_dataframe = add_class_labels(test_dataframe, test_count_ham1, test_count_spam1)

    '''
    train_ham, train_spam, train_count_ham, train_count_spam = read_datasets(train_path_ham[1], train_path_spam[1])
    train_dataframe, bow_vectorizer = bag_of_words_train(train_ham+train_spam, train_count_ham, train_count_spam)
    test_ham, test_spam, test_count_ham, test_count_spam = read_datasets(test_path_ham[1], test_path_spam[1]) #Reading test dataset
    test_dataframe = bag_of_words_test(test_ham+test_spam, test_count_ham, test_count_spam, bow_vectorizer)
    '''
    print("Dataset {}".format(dataset_no+1))
    train_ham, train_spam, train_count_ham, train_count_spam = read_datasets(train_path_ham[dataset_no], train_path_spam[dataset_no])
    train_dataframe, bow_vectorizer = bag_of_words_train(train_ham+train_spam, train_count_ham, train_count_spam)
    test_ham, test_spam, test_count_ham, test_count_spam = read_datasets(test_path_ham[dataset_no], test_path_spam[dataset_no]) #Reading test dataset
    test_dataframe = bag_of_words_test(test_ham+test_spam, test_count_ham, test_count_spam, bow_vectorizer)



    prior,cond_prob = training_multinomialNB(train_dataframe, [0 ,1]) #Class 0 = ham, Class 1 = spam
    predicted = apply_multinomialNB(test_dataframe, test_count_spam+test_count_ham, [0, 1], prior, cond_prob)
    classifier_report(test_dataframe.iloc[:,-1:], predicted)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/



