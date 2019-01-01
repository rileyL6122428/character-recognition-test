import joblib
from data_management.train_subset import flat_images as flat_images_train, number_classes as number_classes_train
from data_management.test_set import flat_images as flat_images_test, number_classes as number_classes_test
from sklearn.metrics import recall_score, precision_score
from classifier_sgd.iteration_2_load_clf import sgd_classifier

predictions_train = sgd_classifier.predict(flat_images_train)
predictions_test = sgd_classifier.predict(flat_images_test)

avg_precision_train = precision_score(
    number_classes_train,
    predictions_train,
    average='macro'
)

avg_precision_test = precision_score(
    number_classes_test,
    predictions_test,
    average='macro'
)

avg_recall_train = recall_score(
    number_classes_train,
    predictions_train,
    average='macro'
)

avg_recall_test = recall_score(
    number_classes_test,
    predictions_test,
    average='macro'
)

print('avg_precision_train = ', avg_precision_train)
# avg_precision_train = 0.7000884561519688
print('avg_recall_train = ', avg_recall_train)
# avg_recall_train =  0.7035864822963217

print('avg_precision_test = ', avg_precision_test)
# avg_precision_test =  0.6890198976018381
print('avg_recall_test = ', avg_recall_test)
# avg_recall_test = 0.6911858974358973
