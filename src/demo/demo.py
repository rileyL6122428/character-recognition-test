from PIL import Image
import numpy
import pdb
from data_visualization.render_character import render_with_pyplot
from classifier_mlp.iteration_2_load_clf import mlp_classifier
from classifier_rfc.iteration_2_load_clf import rfc_classifier
from classifier_sgd.iteration_2_load_clf import sgd_classifier
from data_management.adapters import to_character
from sklearn.metrics import recall_score, precision_score

def as_grayscale_list(rgb_list):
    return [
        255 - ((red + green + blue) / 3)
        for (red, green, blue, alpha)
        in rgb_list
    ]

def as_28_by_28(grayscale_list):
    return numpy.transpose(
        numpy.reshape(grayscale_list, (28, 28))
    )

def flatten(image):
    flattened_image = []

    for row in image:
        for gray_scale_val in row:
            flattened_image.append(gray_scale_val)
            
    return flattened_image

def adapt_for_prediction(img):
    rgb_list = img.getdata()
    grayscale_list = as_grayscale_list(rgb_list)
    grayscale_28_by_28 = as_28_by_28(grayscale_list)
    flattened_image = flatten(grayscale_28_by_28)
    return flattened_image

def predict(img, actual):
    flattened_image = adapt_for_prediction(img)

    mlp_prediction = mlp_classifier.predict([flattened_image])
    rfc_prediction = rfc_classifier.predict([flattened_image])
    sgd_prediction = sgd_classifier.predict([flattened_image])
    
    print('actual = ', actual)
    print('mlp_prediction = ', to_character(mlp_prediction))
    print('rfc_prediction = ', to_character(rfc_prediction))
    print('sgd_prediction = ', to_character(sgd_prediction), '\n')



characters_drawn_by_riley = list(map(
    lambda ascii_number: (chr(ascii_number), chr(ascii_number - 32)), 
    range(97, 122)
))

predictions = []
for (lowercase_chr, uppercase_chr) in characters_drawn_by_riley:
    uppercase_img = Image.open(
        '/Users/rileylittlefield/Desktop/classify_chars_ml/src/demo/riley-'
        + uppercase_chr +
        '-cap.png'
    )

    lowercase_img = Image.open(
        '/Users/rileylittlefield/Desktop/classify_chars_ml/src/demo/riley-' 
        + lowercase_chr +
        '.png'
    )

    predict(uppercase_img, uppercase_chr)
    predict(lowercase_img, lowercase_chr)

images = []
labels = []
for (lowercase_chr, uppercase_chr) in characters_drawn_by_riley:
    # pdb.set_trace()
    images.append(adapt_for_prediction(Image.open(
        '/Users/rileylittlefield/Desktop/classify_chars_ml/src/demo/riley-'
        + uppercase_chr +
        '-cap.png'
    )))
    labels.append(ord(lowercase_chr) - 96)

    images.append(adapt_for_prediction(Image.open(
        '/Users/rileylittlefield/Desktop/classify_chars_ml/src/demo/riley-'
        + lowercase_chr +
        '.png'
    )))
    labels.append(ord(lowercase_chr) - 96)

def grade_performance(characters, labels, classifier, clf_name):
    predictions = classifier.predict(characters)
    precision = precision_score(
        labels,
        predictions,
        average='macro'
    )
    print(clf_name, 'precision = ', precision)

    recall = recall_score(
        labels,
        predictions,
        average='macro'
    )
    print(clf_name,  'recall = ', recall, '\n')


grade_performance(images, labels, mlp_classifier, 'mlp_classifier')
# mlp_classifier precision =  0.6933333333333332
# mlp_classifier recall = 0.7

grade_performance(images, labels, rfc_classifier, 'rfc_classifier')
# rfc_classifier precision =  0.63
# rfc_classifier recall = 0.58

grade_performance(images, labels, sgd_classifier, 'sgd_classifier')
# sgd_classifier precision =  0.43920634920634927
# sgd_classifier recall = 0.42
