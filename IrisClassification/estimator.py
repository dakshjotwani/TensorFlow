from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='train steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    (train_features, train_labels), (test_features, test_labels) = data.load()
    feature_columns = data.get_feature_columns(train_features)

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        n_classes=3)

    classifier.train(
        input_fn=lambda:data.train_input_fn(train_features,
            train_labels,
            args.batch_size),
        steps=args.train_steps)

    evaluation = classifier.evaluate(
        input_fn=lambda:data.eval_input_fn(test_features,
            test_labels,
            args.batch_size))

    print(evaluation)

if __name__ == '__main__':
    tf.app.run(main)
    tf.logging.set_verbosity(tf.logging.INFO)
