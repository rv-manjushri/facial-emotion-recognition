import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import numpy as np

def random_prediction(model, x_test, test_labels):
    index = random.randint(0, len(x_test)-1)
    print("Original Output:", test_labels[index])

    pred = model.predict(x_test[index].reshape(1, 48, 48, 1))
    pred_label = le.inverse_transform([pred.argmax()])[0]
    print("Predicted Output:", pred_label)

    plt.imshow(x_test[index].reshape(48, 48), cmap='gray')
    plt.show()
