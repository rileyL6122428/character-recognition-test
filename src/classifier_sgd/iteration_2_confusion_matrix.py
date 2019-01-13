from sklearn.metrics import confusion_matrix
from data_management.test_set import number_classes as labels, flat_images
from iteration_2_load_clf import sgd_classifier
from data_visualization.character_confusion_matrix import render_confusion_matrix

predictions = sgd_classifier.predict(flat_images)

sgd_confusion_matrix = confusion_matrix(labels, predictions)

# print(sgd_confusion_matrix)
render_confusion_matrix(sgd_confusion_matrix)

# COLS -> LABELS, ROWS -> PREDICTIONS
#     a   ,b   ,c   ,d   ,e   ,f   ,g   ,h   ,i   ,j   ,k   ,l   ,m   ,n   ,o   ,p   ,q   ,r   ,s   ,t   ,u   ,v   ,w   ,x   ,y   ,z   ,
# a   219 ,3   ,7   ,15  ,17  ,7   ,19  ,29  ,0   ,3   ,6   ,1   ,12  ,30  ,26  ,3   ,18  ,1   ,3   ,2   ,29  ,0   ,10  ,10  ,0   ,10  ,
# b   3   ,312 ,4   ,17  ,14  ,0   ,27  ,23  ,6   ,6   ,2   ,8   ,3   ,4   ,9   ,3   ,6   ,1   ,16  ,2   ,1   ,0   ,0   ,2   ,3   ,8   ,
# c   6   ,0   ,389 ,0   ,20  ,3   ,15  ,1   ,0   ,6   ,3   ,3   ,0   ,3   ,14  ,3   ,2   ,3   ,2   ,1   ,2   ,0   ,1   ,0   ,0   ,3   ,
# d   2   ,36  ,3   ,270 ,5   ,1   ,1   ,14  ,1   ,17  ,7   ,13  ,1   ,5   ,51  ,11  ,4   ,0   ,0   ,1   ,13  ,4   ,4   ,1   ,5   ,10  ,
# e   9   ,9   ,36  ,4   ,358 ,11  ,4   ,0   ,1   ,0   ,12  ,1   ,3   ,2   ,6   ,3   ,4   ,5   ,3   ,5   ,0   ,0   ,1   ,2   ,0   ,1   ,
# f   1   ,2   ,0   ,1   ,1   ,340 ,6   ,5   ,2   ,3   ,3   ,5   ,4   ,1   ,0   ,43  ,6   ,16  ,2   ,22  ,0   ,1   ,2   ,2   ,10  ,2   ,
# g   13  ,5   ,16  ,2   ,7   ,6   ,235 ,5   ,0   ,12  ,1   ,0   ,2   ,3   ,8   ,13  ,76  ,2   ,28  ,13  ,9   ,1   ,4   ,3   ,13  ,3   ,
# h   9   ,3   ,1   ,2   ,1   ,3   ,1   ,335 ,11  ,1   ,16  ,20  ,3   ,34  ,0   ,1   ,2   ,1   ,1   ,1   ,10  ,4   ,5   ,10  ,4   ,1   ,
# i   0   ,1   ,0   ,1   ,0   ,3   ,1   ,1   ,273 ,14  ,1   ,121 ,0   ,1   ,0   ,2   ,1   ,1   ,4   ,12  ,2   ,5   ,0   ,13  ,5   ,18  ,
# j   0   ,2   ,6   ,9   ,0   ,7   ,5   ,3   ,12  ,363 ,0   ,17  ,2   ,0   ,3   ,1   ,3   ,1   ,18  ,6   ,5   ,3   ,0   ,1   ,7   ,6   ,
# k   5   ,5   ,10  ,3   ,13  ,3   ,1   ,44  ,4   ,0   ,284 ,15  ,5   ,8   ,0   ,0   ,1   ,14  ,0   ,2   ,7   ,14  ,3   ,31  ,2   ,6   ,
# l   0   ,2   ,11  ,0   ,0   ,0   ,0   ,3   ,129 ,0   ,4   ,275 ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,11  ,8   ,5   ,0   ,8   ,20  ,4   ,
# m   3   ,0   ,0   ,1   ,0   ,1   ,1   ,8   ,0   ,0   ,2   ,0   ,424 ,18  ,0   ,1   ,3   ,3   ,0   ,3   ,5   ,0   ,1   ,1   ,4   ,1   ,
# n   25  ,5   ,1   ,4   ,4   ,0   ,0   ,9   ,1   ,0   ,12  ,0   ,41  ,311 ,3   ,2   ,4   ,1   ,0   ,3   ,15  ,11  ,18  ,4   ,3   ,3   ,
# o   2   ,1   ,8   ,9   ,1   ,0   ,9   ,1   ,1   ,1   ,1   ,0   ,0   ,3   ,433 ,1   ,2   ,2   ,0   ,1   ,1   ,0   ,0   ,1   ,0   ,2   ,
# p   0   ,0   ,1   ,1   ,1   ,12  ,4   ,1   ,3   ,0   ,2   ,1   ,1   ,2   ,3   ,405 ,9   ,11  ,0   ,14  ,1   ,0   ,2   ,1   ,5   ,0   ,
# q   13  ,5   ,4   ,5   ,10  ,15  ,51  ,1   ,2   ,1   ,1   ,0   ,4   ,2   ,10  ,11  ,288 ,1   ,5   ,26  ,15  ,0   ,2   ,2   ,5   ,1   ,
# r   17  ,1   ,13  ,1   ,12  ,11  ,0   ,13  ,0   ,0   ,41  ,0   ,5   ,8   ,1   ,17  ,10  ,274 ,1   ,28  ,0   ,3   ,2   ,14  ,5   ,3   ,
# s   8   ,2   ,4   ,1   ,1   ,4   ,23  ,0   ,3   ,25  ,1   ,0   ,1   ,1   ,2   ,0   ,9   ,1   ,379 ,1   ,4   ,0   ,1   ,3   ,4   ,2   ,
# t   1   ,12  ,5   ,2   ,13  ,44  ,1   ,13  ,11  ,8   ,4   ,8   ,2   ,0   ,1   ,9   ,8   ,36  ,3   ,248 ,0   ,4   ,0   ,1   ,38  ,8   ,
# u   15  ,3   ,2   ,9   ,1   ,0   ,2   ,6   ,0   ,1   ,7   ,1   ,2   ,9   ,14  ,1   ,0   ,2   ,0   ,0   ,370 ,10  ,22  ,1   ,0   ,2   ,
# v   0   ,1   ,0   ,1   ,0   ,1   ,0   ,3   ,0   ,0   ,4   ,4   ,0   ,16  ,0   ,1   ,1   ,20  ,1   ,5   ,27  ,368 ,11  ,5   ,11  ,0   ,
# w   5   ,4   ,0   ,4   ,2   ,3   ,0   ,9   ,0   ,2   ,4   ,2   ,5   ,9   ,0   ,0   ,0   ,1   ,0   ,0   ,20  ,7   ,402 ,1   ,0   ,0   ,
# x   9   ,3   ,0   ,2   ,3   ,3   ,9   ,4   ,5   ,1   ,15  ,6   ,1   ,2   ,0   ,1   ,0   ,4   ,4   ,1   ,0   ,14  ,0   ,355 ,34  ,4   ,
# y   1   ,4   ,0   ,0   ,0   ,4   ,3   ,7   ,4   ,12  ,2   ,9   ,4   ,3   ,0   ,10  ,1   ,7   ,3   ,12  ,1   ,24  ,0   ,14  ,355 ,0   ,
# z   4   ,8   ,3   ,9   ,5   ,1   ,3   ,1   ,4   ,7   ,5   ,9   ,2   ,9   ,1   ,1   ,9   ,0   ,4   ,9   ,3   ,0   ,0   ,21  ,1   ,361 ,