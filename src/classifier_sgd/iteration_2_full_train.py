from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from data_prep.sharpen_images_transformer import SharpenImageTransformer
from data_prep.normalize_grayscale_transformer import NormalizeGrayscaleTransformer
from data_management.train_set import flat_images, number_classes
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib

sgd_classifier = Pipeline([
    ('sharpen_images', SharpenImageTransformer(
        whiten_threshold=50,
        darken_threshold=50
    )),
    ('normalize_grayscale', NormalizeGrayscaleTransformer(
        is_on=True
    )),
    ('sgd_classifier', SGDClassifier(
        random_state=42,
        loss='log',
        alpha=0.001,
        eta0=0.001,
        learning_rate='constant',
        tol=0.0001,
    ))
])

sgd_classifier.fit(flat_images, number_classes)

joblib.dump(
    sgd_classifier,
    '/Users/rileylittlefield/Desktop/classify_chars_ml/src/classifier_sgd/iteration_2.joblib'
)
