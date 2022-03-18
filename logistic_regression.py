import os
import sys
import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

CORRECT_DATASET_NO = [1,2,3]
FEATURE_EXTRACTION_METHOD = ["bag_of_words", "bernoulli"]
CLASSIFIER = ["lr", "sgd"]

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

def sigmoid(z):
    try:
        result = 1/(1+np.exp(-z))
    except:
        print("Overflow occurred for z = {}, result = {}".format(z, result))
    return result

def conditional_prob(train_series, no_of_features, bias, weights):
    sum_wx = 0
    for i in range(no_of_features):
        sum_wx = sum_wx + train_series[i]*weights[i]
    sum_w0_wx = bias + sum_wx
    #print("sum="+str(sum_w0_wx))
    value = sigmoid(sum_w0_wx)
    return value

def partial_derivative(train_split, feature_count, w0, w):
    class_labels = train_split['Class_label'].values.tolist()
    train = train_split.drop("Class_label", axis=1)
    train = train.values.tolist()
    bias_derivative = 0
    error = {}
    derivative = {}
    for example in range(len(train)):
        #prediction = conditional_prob(train.iloc[example], feature_count, w0, w)
        prediction = conditional_prob(train[example], feature_count, w0, w)
        error[example] = class_labels[example] - prediction
        bias_derivative = bias_derivative + error[example]
    for feature in range(len(w)):
        s = 0
        for j in range(len(train)):
            #s = s + train.iat[j, feature]*error[j]
            s = s + train[j][feature] * error[j]
        derivative[feature] = s
    return bias_derivative, derivative

def training_logistic_regression(train_split):
    feature_count = len(train_split.columns) - 1
    w0 = 1 #initialize W
    w = [0]*(feature_count)
    total_iterations = 15
    #ETA = {0.001, 0.003, 0.005, 0.01}
    eta = 0.05
    lambda1 = 3
    for iteration in range(total_iterations):
        #print("Iteration {}>>>>>>>>>>>>>>>>>>>>".format(iteration))
        bias_derivative, derivative = partial_derivative(train_split, feature_count, w0, w)
        w0 = w0 + eta*bias_derivative
        for i in range(len(w)):
            w[i] = w[i] + eta*derivative[i] - (eta*lambda1*w[i])
        #print("bias = {}".format(w0))
        #print("Weights:-")
        #print(w)
        #print("End Iteration {}".format(iteration))
    print("eta = {}".format(eta))
    print("lambda = {}".format(lambda1))
    print("Number of iterations = {}".format(total_iterations))
    return w0, w

def apply_logistic_regression(test_split, w0, w):
    test = test_split.drop("Class_label",axis=1)
    feature_count = len(test.columns)
    test = test.values.tolist()
    test_predictions = []
    for i in range(len(test)):
        value = w0
        for j in range(feature_count):
            value = value + test[i][j]*w[j]
        if value >= 0:
            test_predictions.append(1)
        else:
            test_predictions.append(0)
    return test_predictions



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
    #print(tn, fp, fn, tp)



if __name__ == "__main__":
    try:
        if (str(sys.argv[1]).lower()) in CLASSIFIER and (str(sys.argv[2]).lower()) in FEATURE_EXTRACTION_METHOD and int(sys.argv[3]) in CORRECT_DATASET_NO:
            classifier = str(sys.argv[1]).lower()
            method = str(sys.argv[2]).lower()
            dataset_no = int(sys.argv[3]) - 1
        else:
            raise ValueError("The mentioned classifier, feature extraction method of dataset number was incorrect")
    except ValueError as err:
        print(err)
        print("Please specify the dataset number in the range of [1,3]/Please specify feature extraction method as 'bag_of_words' or 'bernoulli'")
        sys.exit(1)
    print("Dataset {}".format(dataset_no+1))
    if method == "bag_of_words":
        print("Running bag of words")
        train_ham, train_spam, train_count_ham, train_count_spam = read_datasets(train_path_ham[dataset_no], train_path_spam[dataset_no])
        train_dataframe, bow_vectorizer = bag_of_words_train(train_ham + train_spam, train_count_ham, train_count_spam)
        test_ham, test_spam, test_count_ham, test_count_spam = read_datasets(test_path_ham[dataset_no],
                                                                             test_path_spam[dataset_no])  # Reading test dataset
        test_dataframe = bag_of_words_test(test_ham + test_spam, test_count_ham, test_count_spam, bow_vectorizer)
    else:
        print("Running Bernoulli Model")
        train_ham, train_spam, train_count_ham, train_count_spam = read_datasets(train_path_ham[dataset_no],
                                                                                 train_path_spam[dataset_no])
        train_dataframe, bernoulli_vectorizer = bernoulli_train(train_ham + train_spam, train_count_ham,
                                                                train_count_spam)
        test_ham, test_spam, test_count_ham, test_count_spam = read_datasets(test_path_ham[dataset_no], test_path_spam[
            dataset_no])  # Reading test dataset
        test_dataframe = bernoulli_test(test_ham + test_spam, test_count_ham, test_count_spam, bernoulli_vectorizer)

    if classifier == "sgd":
        train_labels = train_dataframe['Class_label'].values.tolist()
        train_X = train_dataframe.drop("Class_label", axis=1).values.tolist()
        test_labels = test_dataframe['Class_label'].values.tolist()
        test_X = test_dataframe.drop("Class_label", axis=1).values.tolist()

        '''
        print("Running SGD Classifier with GridSearchCV")
        sgd_model = SGDClassifier()
        params = {'loss': ["log", "hinge", "squared_hinge"], "penalty": ["l2", "l1", "none"], 'max_iter': [20, 40, 60, 80]}
        grid = GridSearchCV(estimator=sgd_model, param_grid=params)
        grid.fit(train_X, train_labels)
        print("Best score of SGD Classifier with GridSearchCV = {}".format(grid.best_score_))
        print("Best estimator of SGD Classifier obtained from GridSearchCV = {}".format(grid.best_estimator_))
        
        #The above code gave the following output:
        #Running SGD Classifier with GridSearchCV
        #Best score of SGD Classifier with GridSearchCV = 0.9350864890135577
        #Best estimator of SGD Classifier obtained from GridSearchCV = SGDClassifier(loss='log', max_iter=60, penalty='none')
        '''
        print("Running SGD model with best estimator params")
        best_estimator_sgd_model = SGDClassifier(loss='log', max_iter=60, penalty='none')
        best_estimator_sgd_model.fit(train_X, train_labels)
        prediction_labels = best_estimator_sgd_model.predict(test_X)
        classifier_report(test_labels, prediction_labels)

    if classifier == "lr":
        '''
        #The below code was used to split the dataset into respective sizes.
        #Split the dataset into 70% train and 30% validation
        print("Running logistic regression on split dataset")
        train_split, validation_split = train_test_split(train_dataframe, test_size=3/10, shuffle=True, random_state=2)
        print("Length of train_dataframe="+str(len(train_dataframe)))
        print("Length of train_split=" + str(len(train_split)))
        print("Length of validation_split=" + str(len(validation_split)))
        print("Count of spam docs in train= "+str(len(train_split[train_split['Class_label'] == 1])))
        print("Count of ham docs in train= " + str(len(train_split[train_split['Class_label'] == 0])))
        print("Count of spam docs in test= " + str(len(validation_split[validation_split['Class_label'] == 1])))
        print("Count of ham docs in test= " + str(len(validation_split[validation_split['Class_label'] == 0])))



        bias, weights = training_logistic_regression(train_split)
        test_predictions = apply_logistic_regression(validation_split, bias, weights)
        classifier_report(validation_split.iloc[:,-1:], test_predictions)
        '''
        print("Running Logistic Regression")
        bias, weights = training_logistic_regression(train_dataframe)
        test_predictions = apply_logistic_regression(test_dataframe, bias, weights)
        classifier_report(test_dataframe.iloc[:,-1:], test_predictions)

