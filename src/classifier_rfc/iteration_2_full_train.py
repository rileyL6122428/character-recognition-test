from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from data_prep.normalize_grayscale_transformer import NormalizeGrayscaleTransformer
from data_prep.sharpen_images_transformer import SharpenImageTransformer
from data_management.train_set import number_classes, flat_images
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
import joblib

rfc_classifier = Pipeline([
    ('sharpen_image', SharpenImageTransformer(
        darken_threshold=50,
        whiten_threshold=50
    )),
    ('normalize_image', NormalizeGrayscaleTransformer(
        is_on=True
    )),
    ('random_forest', RandomForestClassifier(
        criterion='gini',
        n_estimators=1000
    ))
])

class_predictions = cross_val_predict(
    rfc_classifier,
    flat_images,
    number_classes,
    cv=5,
    n_jobs=2,
    verbose=10
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

print('average precision = ', average_precision) 
print('average recall = ', average_recall)  

joblib.dump(
    rfc_classifier,
    '/Users/rileylittlefield/Desktop/classify_chars_ml/src/classifier_rfc/iteration_2.joblib'
)
