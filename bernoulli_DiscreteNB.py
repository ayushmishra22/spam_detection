import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

CORRECT_DATASET_NO = [1,2,3]
Current_directory = os.getcwd()
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
'''
train_path_ham = ["C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset1\\train\\ham",
                  "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset2\\enron1_train\\enron1\\train\\ham",
                  "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset3\\enron4_train\\enron4\\train\\ham"]
train_path_spam = ["C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset1\\train\\spam",
                   "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset2\\enron1_train\\enron1\\train\\spam",
                   "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset3\\enron4_train\\enron4\\train\\spam"]

test_path_ham = ["C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset1\\test\\ham",
                 "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset2\\enron1_test\\enron1\\test\\ham",
                 "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset3\\enron4_test\\enron4\\test\\ham"]
test_path_spam = ["C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset1\\test\\spam",
                  "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset2\\enron1_test\\enron1\\test\\spam",
                  "C:\\Users\\Ayush\\Desktop\\CS6375\\Project\\dataset3\\enron4_test\\enron4\\test\\spam"]
'''



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
    print("Current Directory = " + os.getcwd())
    cur_dir = os.getcwd()
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

def bernoulli_train(training_data, train_count_ham, train_count_spam):
    bernoulli_vectorizer = CountVectorizer(binary=True)
    vectors = bernoulli_vectorizer.fit_transform(training_data).toarray()
    train_dataframe = create_dataframe(bernoulli_vectorizer.get_feature_names(), vectors)
    train_dataframe = add_class_labels(train_dataframe, train_count_ham, train_count_spam)
    return train_dataframe, bernoulli_vectorizer

def bernoulli_test(test_data, test_count_ham, test_count_spam, vectorizer):
    #test_vectors = vectorizer.transform(test_ham1+test_spam1).toarray()  #Creating test vectors
    test_vectors = vectorizer.transform(test_data).toarray()
    test_dataframe = create_dataframe(vectorizer.get_feature_names(), test_vectors) #Creating test dataframe
    test_dataframe = add_class_labels(test_dataframe, test_count_ham, test_count_spam)
    return test_dataframe

def training_discreteNB(train_dataframe, class_labels):
    total_docs = len(train_dataframe)
    count_docs_in_class = {}
    prior = {}
    nct = {}
    cond_prob = {}
    for i in class_labels:
        count_docs_in_class[i] = len(train_dataframe[train_dataframe["Class_label"] == i])
        prior[i] = count_docs_in_class[i]/total_docs
        dataframe = train_dataframe[train_dataframe["Class_label"] == i]
        dataframe = dataframe.drop(["Class_label"],axis=1)
        nct[i] = dataframe.sum(axis=0)
        for t in dataframe.columns.values.tolist():
            key = str(t)+"|"+str(i)
            cond_prob[key] = (nct[i][t]+1)/(count_docs_in_class[i]+2)
    return prior, cond_prob

def apply_discreteNB(test_dataframe, test_vector_count, class_labels, prior, cond_prob):
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
                    score[k][i] = score[k][i] + np.log(cond_prob[key])
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





if __name__ == '__main__':
    try:
        if int(sys.argv[1]) in CORRECT_DATASET_NO:
            dataset_no = int(sys.argv[1]) - 1
        else:
            raise ValueError("The mentioned dataset number was outside [1,2,3]")
    except ValueError as err:
        print(err)
        print("Please specify the dataset number in the range of [1,3]")
        sys.exit(1)
    print("Running Bernoulli Model")
    print("Dataset {}".format(dataset_no+1))
    train_ham, train_spam, train_count_ham, train_count_spam = read_datasets(train_path_ham[dataset_no], train_path_spam[dataset_no])
    train_dataframe, bernoulli_vectorizer = bernoulli_train(train_ham+train_spam, train_count_ham, train_count_spam)
    test_ham, test_spam, test_count_ham, test_count_spam = read_datasets(test_path_ham[dataset_no], test_path_spam[dataset_no]) #Reading test dataset
    test_dataframe = bernoulli_test(test_ham+test_spam, test_count_ham, test_count_spam, bernoulli_vectorizer)

    prior, cond_prob = training_discreteNB(train_dataframe, [0, 1])
    predicted = apply_discreteNB(test_dataframe, test_count_spam+test_count_ham, [0, 1], prior, cond_prob)
    classifier_report(test_dataframe.iloc[:,-1:], predicted)









