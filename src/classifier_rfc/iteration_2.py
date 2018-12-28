from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from data_prep.normalize_grayscale_transformer import NormalizeGrayscaleTransformer
from data_prep.sharpen_images_transformer import SharpenImageTransformer
from data_management.train_subset import number_classes, flat_images
from classifier_sgd.iteration_2 import f1_score_multi_class

rfc_grid_search = GridSearchCV(
    Pipeline([
        ('image_sharpener', SharpenImageTransformer()),
        ('image_normalizer', NormalizeGrayscaleTransformer()),
        ('rfc_classifier', RandomForestClassifier(random_state=42))
    ]),
    param_grid=[
        {
            'rfc_classifier__n_estimators': [10, 50, 100],
            'rfc_classifier__criterion': ['gini', 'entropy'],
            'image_normalizer__is_on': [True, False],
            'image_sharpener__darken_threshold': [50],
            'image_sharpener__whiten_threshold': [50]
        },
        {
            'rfc_classifier__n_estimators': [10, 50, 100],
            'rfc_classifier__criterion': ['gini', 'entropy'],
            'image_normalizer__is_on': [True, False],
            'image_sharpener__darken_threshold': [None],
            'image_sharpener__whiten_threshold': [None]
        }
    ],
    scoring=f1_score_multi_class,
    cv=3,
    verbose=10
)

rfc_grid_search.fit(flat_images, number_classes)

print('grid_search.best_params_ = ', rfc_grid_search.best_params_)
print('grid_search.best_score_ = ', rfc_grid_search.best_score_)
print('grid_search.cv_results_ = ', rfc_grid_search.cv_results_)
