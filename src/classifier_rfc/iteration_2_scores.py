from classifier_rfc.iteration_2_load_clf import rfc_classifier
from sklearn.metrics import precision_score, recall_score
from data_management.train_set import flat_images, number_classes

predictions = rfc_classifier.predict(flat_images)

avg_precision = precision_score(
    number_classes,
    predictions,
    average='macro'
)

avg_recall = recall_score(
    number_classes,
    predictions,
    average='macro'
)

print('avg_precision = ', avg_precision)
# 0.9999717402362516
print('avg_recall = ', avg_recall)
# 0.9999717194570136
