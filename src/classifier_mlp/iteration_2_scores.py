from classifier_mlp.iteration_2_load_clf import mlp_classifier
from data_management.train_set import number_classes as number_classes_train, flat_images as flat_images_train
from data_management.test_set import number_classes as number_classes_test, flat_images as flat_images_test
from sklearn.metrics import precision_score, recall_score

predictions_train = mlp_classifier.predict(flat_images_train)
predictions_test = mlp_classifier.predict(flat_images_test)

average_precision_train = precision_score (
    number_classes_train,
    predictions_train,
    average='macro'
)

average_precision_test = precision_score (
    number_classes_test,
    predictions_test,
    average='macro'
)

average_recall_train = recall_score(
    number_classes_train,
    predictions_train,
    average='macro'
)

average_recall_test = recall_score(
    number_classes_test,
    predictions_test,
    average='macro'
)

print('average_precision_train = ', average_precision_train)
# average_precision_train =  0.9597057833216434
print('average_recall_train = ', average_recall_train)
# average_recall_train =  0.9595116892911011

print('average_precision_train = ', average_precision_test)
# average_precision_train =  0.9032268031085339
print('average_recall_test = ', average_recall_test)
# average_recall_test =  0.9024839743589742
