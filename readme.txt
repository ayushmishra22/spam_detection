Contact email id: Ayush.Mishra@utdallas.edu
Contact number: +1 469-891-8331

Pre-requisites to run the code:

Import the following python modules:
pip3.9 install os
pip3.9 install sys
pip3.9 install pandas
pip3.9 install numpy
pip3.9 install sklearn
pip3.9 install math


Directory Structure:
Project
   |_main.py
   |_logistic_regression.py
   |_bernoulli_DiscreteNB.py
   |_Dataset1
	|_train
	|_test
   |_Dataset2
	|_enron1_train
	|_enron1_test
   |_Dataset3
	|_enron4_train
	|_enron4_test
   |_readme.txt


How to run the code:
To execute Multinomial Naive Bayes with Bag-of-words representation, run the following command and pass dataset number as parameter
	To run with dataset 1: python.exe main.py 1 
	To run with dataset 2: python.exe main.py 2
	To run with dataset 3: python.exe main.py 3


To execute Discrete Naive Bayes with Bernoulli representation, run the following command and pass dataset number as parameter:
	To run with dataset 1: python.exe bernoulli_DiscreteNB.py 1
	To run with dataset 2: python.exe bernoulli_DiscreteNB.py 2
	To run with dataset 3: python.exe bernoulli_DiscreteNB.py 3


To execute Logistic Regression, run the following command. Parameters to be passed: lr/sgd bag_of_words/bernoulli 1/2/3 (classifier, representation, dataset number)
	To run LR with bag_of_words on dataset 1: python.exe logistic_regression.py lr bag_of_words 1
	To run LR with bag_of_words on dataset 2: python.exe logistic_regression.py lr bag_of_words 2
	To run LR with bag_of_words on dataset 3: python.exe logistic_regression.py lr bag_of_words 3

	To run LR with bernoulli on dataset 1: python.exe logistic_regression.py lr bernoulli 1
	To run LR with bernoulli on dataset 2: python.exe logistic_regression.py lr bernoulli 2
	To run LR with bernoulli on dataset 3: python.exe logistic_regression.py lr bernoulli 3

	To run SGDClassifier with bag_of_words on dataset 1: python.exe logistic_regression.py sgd bag_of_words 1
	To run SGDClassifier with bag_of_words on dataset 2: python.exe logistic_regression.py sgd bag_of_words 2
	To run SGDClassifier with bag_of_words on dataset 3: python.exe logistic_regression.py sgd bag_of_words 3

	To run SGDClassifier with bernoulli on dataset 1: python.exe logistic_regression.py sgd bernoulli 1
	To run SGDClassifier with bernoulli on dataset 2: python.exe logistic_regression.py sgd bernoulli 2
	To run SGDClassifier with bernoulli on dataset 3: python.exe logistic_regression.py sgd bernoulli 3