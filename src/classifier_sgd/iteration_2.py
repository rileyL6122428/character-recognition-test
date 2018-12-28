from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from data_prep.sharpen_images_transformer import SharpenImageTransformer
from data_prep.normalize_grayscale_transformer import NormalizeGrayscaleTransformer
from data_management.train_subset import flat_images, number_classes
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import pdb
from functools import partial

def f1_score_multi_class(estimator, X, y):
    predictions = estimator.predict(X)

    return f1_score(
        y,
        predictions,
        average='macro'
    )


grid_search = GridSearchCV(
    Pipeline([
        ('sharpen_images', SharpenImageTransformer(whiten_threshold=50, darken_threshold=50)),
        ('normalize_grayscale', NormalizeGrayscaleTransformer()),
        ('sgd_classifier', SGDClassifier(random_state=42, loss='log'))
    ]),
    param_grid=[
        {
            'sgd_classifier__alpha': [0.001, 0.0001],
            'sgd_classifier__tol': [0.001, 0.0001],
            'sgd_classifier__learning_rate': ['optimal'],
            'sharpen_images__whiten_threshold': [None],
            'sharpen_images__darken_threshold': [None],
            'normalize_grayscale__is_on': [True, False]
        },

        {
            'sgd_classifier__alpha': [0.001, 0.0001],
            'sgd_classifier__tol': [0.001, 0.0001],
            'sgd_classifier__learning_rate': ['optimal'],
            'sharpen_images__whiten_threshold': [50],
            'sharpen_images__darken_threshold': [50],
            'normalize_grayscale__is_on': [True, False]
        },
        {
            'sgd_classifier__alpha': [0.001, 0.0001],
            'sgd_classifier__tol': [0.001, 0.0001],
            'sgd_classifier__learning_rate': ['constant'],
            'sgd_classifier__eta0': [0.1, 0.001],
            'sharpen_images__whiten_threshold': [None],
            'sharpen_images__darken_threshold': [None],
            'normalize_grayscale__is_on': [True, False]
        },

        {
            'sgd_classifier__alpha': [0.001, 0.0001],
            'sgd_classifier__tol': [0.001, 0.0001],
            'sgd_classifier__learning_rate': ['constant'],
            'sgd_classifier__eta0': [0.1, 0.001],
            'sharpen_images__whiten_threshold': [50],
            'sharpen_images__darken_threshold': [50],
            'normalize_grayscale__is_on': [True, False]
        },
    ],
    scoring=f1_score_multi_class,
    cv=3,
    verbose=10
)

grid_search.fit(flat_images, number_classes)

print('best_params_ = ', grid_search.best_params_)
# best_params_ (on train subset) = { 
#     'normalize_grayscale__is_on': True, 
#     'sgd_classifier__alpha': 0.001, 
#     'sgd_classifier__eta0': 0.001,
#     'sgd_classifier__learning_rate': 'constant',
#     'sgd_classifier__tol': 0.0001,
#     'sharpen_images__darken_threshold': None,
#     'sharpen_images__whiten_threshold': None
# } 

print('best_score_ = ', grid_search.best_score_)
# best_score_ (f1, train subset) 0.6377849057306804
print('cv_reults', grid_search.cv_results_)
