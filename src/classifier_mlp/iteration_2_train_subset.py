from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from data_prep.sharpen_images_transformer import SharpenImageTransformer
from data_prep.normalize_grayscale_transformer import NormalizeGrayscaleTransformer
from data_management.train_subset import flat_images, number_classes
from scoring.f1_for_grid_sv import f1_score_multi_class

mlp_grid_search = GridSearchCV(
    Pipeline([
        ('image_sharpener', SharpenImageTransformer()),
        ('image_normalizer', NormalizeGrayscaleTransformer(is_on=True)),
        ('mlp_classifier', MLPClassifier(random_state=42))
    ]),
    param_grid=[
        {
            'mlp_classifier__hidden_layer_sizes': [(783, 28), (28,)],
            'mlp_classifier__activation': ['logistic', 'relu'],
            'mlp_classifier__solver': ['lbfgs', 'adam'],
            'mlp_classifier__learning_rate': ['constant', 'adaptive'],
            'mlp_classifier__learning_rate_init': [0.01],
            'image_sharpener__whiten_threshold': [50],
            'image_sharpener__darken_threshold': [50],
        },
        {
            'mlp_classifier__hidden_layer_sizes': [(783, 28), (28,)],
            'mlp_classifier__activation': ['logistic', 'relu'],
            'mlp_classifier__solver': ['lbfgs', 'adam'],
            'mlp_classifier__learning_rate': ['constant', 'adaptive'],
            'mlp_classifier__learning_rate_init': [0.01],
            'image_sharpener__whiten_threshold': [None],
            'image_sharpener__darken_threshold': [None],
        },
    ],
    scoring=f1_score_multi_class,
    cv=3,
    verbose=10,
    n_jobs=2
)

mlp_grid_search.fit(flat_images, number_classes)

print('grid_search.best_params_ = ', mlp_grid_search.best_params_)
# {
#   'image_sharpener__darken_threshold': None,
#   'image_sharpener__whiten_threshold': None,
#   'mlp_classifier__activation': 'relu',
#   'mlp_classifier__hidden_layer_sizes': (783, 28),
#   'mlp_classifier__learning_rate': 'constant',
#   'mlp_classifier__learning_rate_init': 0.01,
#   'mlp_classifier__solver': 'lbfgs'
# }
print('grid_search.best_score_ = ', mlp_grid_search.best_score_)
# F1_score = 0.726795811620147
print('grid_search.cv_results_ =', mlp_grid_search.cv_results_)
