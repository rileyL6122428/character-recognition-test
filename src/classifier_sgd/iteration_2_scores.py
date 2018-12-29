import joblib
from data_management.train_subset import flat_images, number_classes
from sklearn.metrics import recall_score, precision_score
from classifier_sgd.iteration_2_load_clf import sgd_classifier

predictions = sgd_classifier.predict(flat_images)

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
0.7000884561519688
print('avg_recall = ', avg_recall)
0.7035864822963217
