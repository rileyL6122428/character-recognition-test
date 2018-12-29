from sklearn.metrics import f1_score

def f1_score_multi_class(estimator, X, y):
    predictions = estimator.predict(X)

    return f1_score(
        y,
        predictions,
        average='macro'
    )
