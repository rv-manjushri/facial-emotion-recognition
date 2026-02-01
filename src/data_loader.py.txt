import os
import pandas as pd

def load_dataset(directory):
    image_paths = []
    labels = []

    for label in os.listdir(directory):
        for filename in os.listdir(directory + label):
            image_path = os.path.join(directory, label, filename)
            image_paths.append(image_path)
            labels.append(label)
        print(label, "Completed")

    return image_paths, labels

def create_dataframes(train_dir, test_dir):
    train = pd.DataFrame()
    test  = pd.DataFrame()

    train['image'], train['label'] = load_dataset(train_dir)
    train = train.sample(frac=1).reset_index(drop=True)

    test['image'], test['label'] = load_dataset(test_dir)
    return train, test
