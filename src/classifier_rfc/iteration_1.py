from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from data_management.train_set import flat_images, number_classes

forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)

class_predictions = cross_val_predict(
    forest_clf,
    flat_images,
    number_classes,
    cv=5
)

average_recall = recall_score(
    number_classes,
    class_predictions,
    average='macro'
)

average_precision = precision_score(
    number_classes,
    class_predictions,
    average='macro'
)

print('average precision = ', average_precision)  # 0.8249868333239829
print('average recall = ', average_recall)  # 0.823133484162896
