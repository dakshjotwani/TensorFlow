import tensorflow as tf
import pandas as pd

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

SPECIES = ['Sentosa', 'Versicolor', 'Virginica']

def download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    
    return train_path, test_path

def load(labels_name='Species'):
    train_path, test_path = download()
    
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(labels_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(labels_name)
    
    return (train_x, train_y), (test_x, test_y)

def get_feature_columns(train_x):
    feature_columns = []
    for column_key in train_x.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=column_key))
    return feature_columns

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(buffer_size=1000).repeat().batch(batch_size)
    return dataset

def eval_input_fn(features, labels, batch_size):
    features = dict(features)

    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "Batch_size cannot be None"
    dataset = dataset.batch(batch_size)

    return dataset
