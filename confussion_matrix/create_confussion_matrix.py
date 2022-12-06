import re
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

with open('output.txt', 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    classes = ['a', 'b', 'c', 'd', 'e', 'f']
    y_pred = []
    y_actual = []
    index_image = 0
    for line in lines:
        # Split the line for every whitespace
        line = re.split(r'\s+', line)
        # Remove empty strings
        line = [x for x in line if x]

        # Remove all elements that are '--'
        line = [x for x in line if x != '--']

        # Convert all values into numerical type
        line = [float(x) for x in line]

        # Create a new array with all values that are integers
        expected_classes = [x for x in line if x.is_integer()]

        # Create a new array with all values that are floats
        predicted_classes = [x for x in line if not x.is_integer()]

        # Get the index of the highest value in the predicted_classes array
        predicted_class = predicted_classes.index(max(predicted_classes))

        # Get the index of the highest value in the expected_classes array
        expected_class = expected_classes.index(max(expected_classes))

        print(
            f"Image {index_image} is predicted as {classes[predicted_class]} and is actually {classes[expected_class]}")
        index_image += 1

        y_pred.append(classes[predicted_class])
        y_actual.append(classes[expected_class])

    # Create a confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_actual, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
    cm_display.plot()
    plt.show()
