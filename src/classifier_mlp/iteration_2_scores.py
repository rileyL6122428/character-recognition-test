from classifier_mlp.iteration_2_load_clf import mlp_classifier
from data_management.train_set import number_classes, flat_images
from sklearn.metrics import precision_score, recall_score

predictions = mlp_classifier.predict(flat_images)

average_precision = precision_score (
    number_classes,
    predictions,
    average='macro'
)

average_recall = recall_score(
    number_classes,
    predictions,
    average='macro'
)

print('average_precision = ', average_precision)
# average_precision =  0.9597057833216434
print('average_recall = ', average_recall)
# average_recall =  0.9595116892911011
