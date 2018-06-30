import time
import json
import tensorflow as tf
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion


def apply_majority_voting(data, majority_class):
    data_labels = []
    for i in range(len(data)):
        data_labels.append(majority_class)
    return data, data_labels


def write_result_file(data_id, data_labels):
    result = "result.json"

    data = {}
    for i in range(len(data_id)):
        # data[str(data_id[i])] = str(data_labels[i])
        if data_labels[i].__eq__(0):  # [0, 0, 0, 1]
            data[str(data_id[i])] = "support"
        if data_labels[i].__eq__(1):  # [0, 0, 1, 0]
            data[str(data_id[i])] = "deny"
        if data_labels[i].__eq__(2):  # [0, 1, 0, 0]
            data[str(data_id[i])] = "query"
        if data_labels[i].__eq__(3):  # [1, 0, 0, 0]
            data[str(data_id[i])] = "comment"
    with open(result, 'w') as outfile:
        json.dump(data, outfile)
    print("Result file written")


file_train = "C:/Users/Becci/Documents/Uni_Mannheim/Thesis/03_Data/Working/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json"
file_dev = "C:/Users/Becci/Documents/Uni_Mannheim/Thesis/03_Data/Working/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json"

start_time = time.time()

x_train, y_train, id_train = processData.load_data(file_train)
x_dev, y_dev, id_dev = processData.load_data(file_dev)

'''
# Majority Voting
majority_class = processData.get_majority_class(y_dev)
x_test_mv, y_test_mv = apply_majority_voting(id_dev, majority_class)
write_result_file(x_test_mv, y_test_mv)
'''


load_time = time.time()
load_time = load_time - start_time
print("Load Time: " + str(load_time))

# Feature Extraction
'''
transformer = processData.PeriodTransformer()
x_train_svm = transformer.fit_transform(x_train)
x_dev_svm = transformer.transform(x_dev)
'''

modVariable = 0

# tt = processData.TfidfTransformer()
# pt = processData.PeriodTransformer()
# et = processData.ExclamationMarkTransformer()
# qt = processData.QuestionMarkTransformer()
# ct = processData.CapitalRatioTransformer()
# nt = processData.NegativeWordsTransformer()
# swt = processData.SwearWordsTransformer()
# st = processData.SourceTweetTransformer()

# combined_features = FeatureUnion([('tt', processData.TfidfTransformer()),
#                                   ('pt', processData.PeriodTransformer()),
#                                   ('et', processData.ExclamationMarkTransformer()),
#                                   ('qt', processData.QuestionMarkTransformer()),
#                                   ('ct', processData.CapitalRatioTransformer()),
#                                   ('nt', processData.NegativeWordsTransformer()),
#                                   ('swt', processData.SwearWordsTransformer())])
                                  # ('w2v', processData.Word2Vec())])
                                  # ('st', processData.SourceTweetTransformer())])

# combined_features = FeatureUnion([('pt', processData.PeriodTransformer()),
#                                   ('et', processData.ExclamationMarkTransformer())])

combined_features = processData.Word2Vec()

x_train_svm = combined_features.fit_transform(id_train)
x_dev_svm = combined_features.transform(id_dev)

feature_extraction_time = time.time()
feature_extraction_time = feature_extraction_time - start_time
print("Feature Extraction Time: " + str(feature_extraction_time))

clf = OneVsRestClassifier(SVC(C=1.5, kernel='linear', probability=True))
clf.fit(x_train_svm, y_train)

fit_time = time.time()
fit_time = fit_time - start_time
print("Fit Time: " + str(fit_time))

predictions = clf.predict(x_dev_svm)

prediction_time = time.time()
prediction_time = prediction_time - start_time
print("Prediction Time: " + str(prediction_time))

write_result_file(id_dev, predictions)

stop_time = time.time()
operations_time = stop_time - start_time
print("Done, Operation Time: " + str(operations_time))
