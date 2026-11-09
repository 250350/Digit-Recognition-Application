"""Train the MNIST MLP and save its model and evaluation results."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('KERAS_HOME', str(ROOT / '.cache/keras'))
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import json
from datetime import datetime, timezone

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

SEED = 42


def save_results(model, history, stop, evaluations, images, labels):
    """Save the test confusion matrix and the first twenty errors for inspection."""
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    probabilities = model.predict(images.astype('float32') / 255, batch_size=128, verbose=0)
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (labels, predicted), 1)
    wrong = np.flatnonzero(predicted != labels)
    examples = [
        {'test_index': int(i), 'true': int(labels[i]), 'predicted': int(predicted[i]),
         'confidence': float(probabilities[i, predicted[i]])}
        for i in wrong[:20]
    ]

    grid = np.zeros((520, 600), np.uint8)
    for n, example in enumerate(examples):
        y, x = (n // 5) * 130, (n % 5) * 120
        grid[y:y + 84, x:x + 84] = cv2.resize(images[example['test_index']], (84, 84))
        cv2.putText(grid, f"{example['true']} -> {example['predicted']}", (x + 2, y + 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.putText(grid, f"index {example['test_index']}", (x + 2, y + 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
    cv2.imwrite(str(results / 'misclassified.png'), grid)
    with (results / 'confusion_matrix.csv').open('w') as file:
        file.write('true/predicted,' + ','.join(map(str, range(10))) + '\n')
        for digit, row in enumerate(confusion):
            file.write(str(digit) + ',' + ','.join(map(str, row)) + '\n')

    metrics = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'configuration': {
            'dataset': 'keras.datasets.mnist', 'seed': SEED,
            'split': 'default_rng(42) permutation: first 50000 train, remaining 10000 validation; official 10000 test',
            'pixels': 'white on black; float32 /255; (N,28,28)',
            'labels': 'integer 0-9; sparse_categorical_crossentropy',
            'architecture': ['Input(28,28)', 'Flatten', 'Dense(128,relu)', 'Dense(64,relu)', 'Dense(10,softmax)'],
            'parameters': model.count_params(), 'optimizer': 'Adam', 'learning_rate': 0.001,
            'batch_size': 128, 'max_epochs': 20,
            'early_stopping': {'monitor': 'val_loss', 'patience': 3, 'min_delta': 0.0001,
                               'restore_best_weights': True},
            'deterministic_ops': True, 'threads': 2,
        },
        'environment': {'tensorflow': tf.__version__, 'keras': keras.__version__, 'numpy': np.__version__},
        'epochs_completed': len(history.epoch), 'best_epoch': int(stop.best_epoch + 1),
        'history': history.history, 'evaluation_restored_best_weights': evaluations,
        'test_accuracy': evaluations['test']['accuracy'], 'test_loss': evaluations['test']['loss'],
        'test_error_count': len(wrong),
        'confusion_matrix_rows_true_columns_predicted': confusion.tolist(),
        'misclassified_examples': examples,
    }
    (results / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(evaluations, indent=2))


def main():
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)

    (images, labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    # Only the original training pool is split; the official test set stays separate.
    order = np.random.default_rng(SEED).permutation(len(images))
    train_ids, val_ids = order[:50000], order[50000:]
    x_train = images[train_ids].astype('float32') / 255
    x_val = images[val_ids].astype('float32') / 255
    x_test = test_images.astype('float32') / 255
    y_train, y_val = labels[train_ids], labels[val_ids]

    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ], name='mnist_ann')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001,
                                         restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20,
                        batch_size=128, callbacks=[stop], verbose=2)

    evaluations = {
        'train': model.evaluate(x_train, y_train, verbose=0, return_dict=True),
        'validation': model.evaluate(x_val, y_val, verbose=0, return_dict=True),
        'test': model.evaluate(x_test, test_labels, verbose=0, return_dict=True),
    }
    path = ROOT / 'app/models/mnist_ann.keras'
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    save_results(model, history, stop, evaluations, test_images, test_labels)


if __name__ == '__main__':
    main()
