from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from data_prep.sharpen_images_transformer import SharpenImageTransformer
from data_prep.normalize_grayscale_transformer import NormalizeGrayscaleTransformer
from classifier_sgd.iteration_2 import f1_score_multi_class
from data_management.train_subset import flat_images, number_classes

mlp_grid_search = GridSearchCV(
    Pipeline([
        ('image_sharpener', SharpenImageTransformer()),
        ('image_normalizer', NormalizeGrayscaleTransformer(is_on=True)),
        ('mlp_classifier', MLPClassifier(random_state=42))
    ]),
    param_grid=[
        {
            'image_sharpener__whiten_threshold': [50],
            'image_sharpener__darken_threshold': [50],
        },
        {
            'image_sharpener__whiten_threshold': [None],
            'image_sharpener__darken_threshold': [None],
        },
    ],
    scoring=f1_score_multi_class,
    cv=3,
    verbose=10
)

mlp_grid_search.fit(flat_images, number_classes)

print('grid_search.best_params_ = ', mlp_grid_search.best_params_)
print('grid_search.best_score_ = ', mlp_grid_search.best_score_)
print('grid_search.cv_results_ =', mlp_grid_search.cv_results_)
