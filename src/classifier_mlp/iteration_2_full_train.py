from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from data_management.train_set import number_classes, flat_images
from data_prep.normalize_grayscale_transformer import NormalizeGrayscaleTransformer
from data_prep.sharpen_images_transformer import SharpenImageTransformer
import joblib

mlp_classifier = Pipeline([
    ('image_sharpener', SharpenImageTransformer(
        darken_threshold=None,
        whiten_threshold=None
    )),
    ('grayscale normalizer', NormalizeGrayscaleTransformer(
        is_on=True
    )),
    ('mlp_classifier', MLPClassifier(
        random_state=42,
        activation='relu',
        hidden_layer_sizes=(783, 28),
        learning_rate='constant',
        learning_rate_init=0.01,
        solver='lbfgs',
    ))
])

mlp_classifier.fit(flat_images, number_classes)

joblib.dump(
    mlp_classifier,
    '/Users/rileylittlefield/Desktop/classify_chars_ml/src/classifier_mlp/iteration_2.joblib'
)
