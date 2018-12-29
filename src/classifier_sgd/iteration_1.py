from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from data_management.train_set import flat_images, number_classes, images_28_by_28, number_classes
from data_management.adapters import to_character
from data_visualization.render_character import render_in_terminal
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sgd_classifier = SGDClassifier(random_state=42)

class_predictions = cross_val_predict(
    sgd_classifier,
    flat_images,
    y=number_classes,
    cv=5
)

avg_precision_score = precision_score(number_classes, class_predictions, average='macro')
avg_recall_score = recall_score(number_classes, class_predictions, average='macro')

print('STATS WITH NO HYPER PARAMETER TUNING')
print('* 5 Cross Validation Folds')
print('* avg precision score = ', avg_precision_score)  # 0.5584353509972628
print('* avg recall score = ', avg_recall_score)  # 0.528393665158371
# f1_score = 0.54299930804
