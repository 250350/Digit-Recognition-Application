# Handwritten Digit Recognition

A small Flask application that recognizes one handwritten digit from 0 to 9. The included Keras MLP was trained on MNIST and achieved **97.15% test accuracy**. Starting Flask loads the saved model; it does not train or download MNIST.

## Install and run

Tested on Windows 11 with Python 3.13.14 (64-bit), TensorFlow 2.21.0 and Keras 3.15.1, using CPU. In PowerShell, from the project directory:

```powershell
# Create only if you do not already have .venv:
py -3.13 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe wsgi.py
```

Open http://127.0.0.1:5000. Draw one white digit on the black canvas and press **Recognize**. The ten bars show softmax scores on a linear 0-100% scale; the highest is highlighted. Scores are not a guarantee of correctness. **Clear** resets the canvas and scores. Stop the local demonstration server with Ctrl+C. No internet is needed after installation to use the included model.

## Train again if needed

```powershell
& .\.venv\Scripts\python.exe create_mnist_model.py
```

The first training run downloads MNIST into `.cache/keras`. It replaces `app/models/mnist_ann.keras` and creates a local `results/` folder with metrics, a confusion matrix and examples of mistakes. Restart Flask after replacing the model. Training is not needed for the live demonstration.

The architecture is `Input(28,28) -> Flatten -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(10, softmax)` (109,386 parameters). Two moderate hidden layers keep training practical on CPU. Pixels are `float32 / 255`; labels are integers 0-9. A seeded NumPy permutation splits the original 60,000 training images into 50,000 training and 10,000 validation images. The official 10,000 test images remain separate.

Training uses seed 42, deterministic TensorFlow operations, two inter/intra-op threads, Adam (learning rate 0.001), sparse categorical cross-entropy, batch size 128 and at most 20 epochs. Early stopping monitors validation loss with patience 3 and min_delta 0.0001, restoring the best weights. The recorded experiment completed 8 epochs and selected epoch 5. Code cleanup retained these settings; the existing model was not retrained. Exact numerical reproducibility across other platforms and dependency versions is not guaranteed.

| Set | Accuracy | Loss |
|---|---:|---:|
| Training | 98.51% | 0.048553 |
| Validation | 96.88% | 0.100980 |
| Test | 97.15% | 0.093391 |

These are evaluations of the restored best weights. The test set contained 285 errors. The report embeds the results and confusion matrix, so local working files are not required to read it.

## Code and input preparation

- `create_mnist_model.py`: training, evaluation and result export.
- `wsgi.py`: local entry point.
- `app/main.py`: load the model once, serve GET and handle prediction POST.
- `app/preprocessing.py`: validate PNG/base64, reject empty input, crop, scale the longest side to 20 pixels and centre the mass on a 28 x 28 background. Return `float32 (1,28,28)` in [0,1].
- `app/templates/drawing.html`: English mouse/touch canvas and ten softmax bars.
- `app/models/mnist_ann.keras`: saved trained model.
- `report/Handwritten_Digit_Recognition_Report.docx`: method, results and limitations.

Incorrect or empty input produces an English error message. Handwriting, rotation and scribbles can differ substantially from MNIST, and the model can be confidently wrong. The server is for a local classroom demonstration. Before presenting, check drawing, Clear, empty input and a zero on the intended device; physical touch support has not been verified.

## Sources and adaptations

1. [agnivahp demonstration repository](https://github.com/agnivahp/Tensorflow-Handwritten-digit-recognition-demo).
2. [Google Cloud MNIST tutorial](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/tree/master/tensorflow-mnist-tutorial).
3. [Philip Purwoko, How To Deploy Digit Recognition Model Into Drawing App](https://medium.com/analytics-vidhya/how-to-deploy-digit-recognition-model-into-drawing-app-6e59f82a199c).

Author examples: [training](https://gist.github.com/PhilipPurwoko/1f87a9599d92e090f35c9094f90d000a), [Flask](https://gist.github.com/PhilipPurwoko/fce7d076fc008cf9b6415e05487434d8), [drawing interface](https://gist.github.com/PhilipPurwoko/ed1194b9b1308599d8bc6ba5c53e0ed8), [WSGI](https://gist.github.com/PhilipPurwoko/99f892a94866396178db0c8a47873c37).

The sources inspired the MNIST exercise and canvas-to-Flask flow. This implementation uses two hidden layers, normalized and centred input, validation and early stopping, the `.keras` format, input error handling and all ten softmax scores. The metrics above come from this project's experiment, not from the source authors.
