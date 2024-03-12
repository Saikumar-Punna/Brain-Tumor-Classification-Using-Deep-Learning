
# Brain Tumor Classification Using Deep Learning

This project utilizes deep learning to classify brain tumor images into two categories: "No Brain Tumor" and "Yes Brain Tumor." The model is trained on a dataset of brain images, and the classification is done through a Flask web application.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Brain tumors are a critical health issue, and early detection is crucial for effective treatment. This project aims to provide a simple tool for classifying brain tumor images using deep learning techniques.

## Prerequisites

- Python 3.10.13
- TensorFlow
- Keras
- Flask
- OpenCV
- PIL (Pillow)

## Installation

1. Clone the repository:
   git clone https://github.com/Saikumar-Punna/Brain-Tumor-Classification-Deep-Learning.git
   cd Brain-Tumor-Classification-Deep-Learning

2. Install dependencies:
   pip install -r requirements.txt

## Usage

1. Run the Flask application:
   python app.py

2. Open your web browser and navigate to [http://localhost:5000/](http://localhost:5000/).

3. Upload brain tumor images and use the web interface to classify them.

## Model Training

The model is trained on brain tumor images. Refer to the `model_training.ipynb` notebook for details on the model architecture and training process.

## Deployment

The Flask application serves as the deployment platform. It provides a web interface for users to upload brain tumor images and receive classification results.

## Contributing

Contributions are welcome! If you have ideas for improvements or find issues, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
