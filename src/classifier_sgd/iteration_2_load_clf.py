import joblib
from data_management.train_subset import flat_images, number_classes
from sklearn.metrics import recall_score, precision_score
import pdb

sgd_classifier = joblib.load(
    '/Users/rileylittlefield/Desktop/classify_chars_ml/src/classifier_sgd/iteration_2.joblib'
)