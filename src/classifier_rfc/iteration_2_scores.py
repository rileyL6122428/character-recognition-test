from classifier_rfc.iteration_2_load_clf import rfc_classifier
from sklearn.metrics import precision_score, recall_score
from data_management.train_set import flat_images as flat_images_train, number_classes as number_classes_train
from data_management.test_set import flat_images as flat_images_test, number_classes as number_classes_test

predictions_train = rfc_classifier.predict(flat_images_train)
predictions_test = rfc_classifier.predict(flat_images_test)

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
# avg_precision_train = 0.9999717402362516
print('avg_recall_train = ', avg_recall_train)
# avg_recall_train = 0.9999717194570136

print('avg_precision_test = ', avg_precision_test)
# avg_precision_test =  0.8825953227631799
print('avg_recall_test = ', avg_recall_test)
# avg_recall_test =  0.882051282051282
