from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score, precision_score
from data_management.train_set import flat_images, number_classes

mlp_classifier = MLPClassifier(hidden_layer_sizes=(784, 28, 1), activation='relu')
# mlp_classifier = MLPClassifier(hidden_layer_sizes=(784, 1), activation='relu')

class_predictions = cross_val_predict(
    mlp_classifier,
    flat_images,
    number_classes,
    cv=5
)

average_precision = precision_score(
    number_classes,
    class_predictions,
    average='macro'
)

average_recall = recall_score(
    number_classes,
    class_predictions,
    average='macro'
)

print('average_precision = ', average_precision)  # 0.007396449704142012
print('average_recall = ', average_recall)  # 0.038461538461538464
