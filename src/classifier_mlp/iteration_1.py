from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score, precision_score
from data_management.train_set import flat_images, number_classes
from data_management.adapters import normalize_image_0_to_1

normalized_images = normalize_image_0_to_1(flat_images)

print('images normalized')

mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(784, 28),
    activation='relu',
    alpha=0.01,
    max_iter=200,
    learning_rate='constant',
    learning_rate_init=0.0001
)

class_predictions = cross_val_predict(
    mlp_classifier,
    normalized_images,
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

print('average_precision = ', average_precision)  # 0.787554 (training subset)
print('average_recall = ', average_recall)  # 0.784897 (training subset)
