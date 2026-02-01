from data_loader import create_dataframes
from feature_extractor import extract_features
from model_builder import build_model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

TRAIN_DIR = "../data/train/"
TEST_DIR  = "../data/test/"

train, test = create_dataframes(TRAIN_DIR, TEST_DIR)

train_features = extract_features(train['image'])
test_features  = extract_features(test['image'])

x_train = train_features / 255.0
x_test  = test_features  / 255.0

le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_test  = le.transform(test['label'])

y_train = to_categorical(y_train, num_classes=7)
y_test  = to_categorical(y_test, num_classes=7)

model = build_model()
history = model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test))
