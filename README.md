# Digit Recognition Application

A web application that recognizes handwritten digits (0–9) using an Artificial Neural Network (ANN) trained on the MNIST dataset.

The application provides a drawing canvas where users can draw a digit and submit it to the trained neural network for prediction.

## Features

- Interactive drawing canvas.
- Digit recognition using a pre-trained ANN model.
- Image preprocessing using OpenCV.
- Real-time prediction displayed in the web interface.
- Clear canvas, drawing and eraser tools.

## Technologies

- Python
- TensorFlow / Keras
- Flask
- NumPy
- OpenCV
- HTML, CSS and JavaScript
- Bootstrap

## Requirements

- Python installed on your computer.
- pip package manager.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Digit-Recognition-Application.git
```

Navigate to the project directory:

```bash
cd Digit-Recognition-Application
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install flask tensorflow numpy opencv-python
```

### 4. Run the application

Start the Flask development server:

```bash
flask --app wsgi run
```
or just:
```bash
flask run
```

### 5. Open the application

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

## How to Use

1. Open the application in your browser.
2. Draw a handwritten digit (0–9) on the canvas.
3. Click **Detect Number**.
4. The application processes the image and passes it to the trained ANN model.
5. The predicted digit is displayed on the screen.

Use the **Clear** button to erase the canvas, or the **Eraser** tool to remove parts of your drawing.

## Machine Learning Model

The application uses a pre-trained Artificial Neural Network trained on the MNIST handwritten digit dataset.

The model is stored in:

```text
app/mnist_classification.h5
```

The model achieved approximately **97% classification accuracy** on the MNIST test dataset containing 10,000 images.

## Notes

- The pre-trained model must be present in the `app/` directory.
- The application expects handwritten digits as input.
- The model processes images resized to 28 × 28 pixels.
- The project is intended for educational purposes and demonstrates the deployment of a trained neural network in a Flask web application.
