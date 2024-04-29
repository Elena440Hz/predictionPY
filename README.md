# Diabetes Prediction Model

This project implements a machine learning model to predict the likelihood of diabetes based on certain medical measurements. The model is trained on the Pima Indians Diabetes Dataset.
It was so hard thank goodness youtube video exists and also it was thankfully six steps. 

## Usage

To use the diabetes prediction model:

1. Clone the repository:
   ```bash
   git clone https://github.com/Elena440Hz/diabetes-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd diabetes-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the `predict_diabetes.py` script:
   ```bash
   python predict_diabetes.py
   ```

## Dataset

The dataset used for training the model is the [Pima Indians Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes). It contains various medical measurements for a group of Pima Indian women, along with a binary outcome indicating whether each individual has diabetes or not.

## Model Training

The machine learning model is trained using the Support Vector Machine (SVM) algorithm. The data is preprocessed, and features are standardized before training the model.

## Evaluation

The trained model is evaluated using accuracy and a classification report. The accuracy metric indicates the overall performance of the model, while the classification report provides detailed metrics such as precision, recall, and F1-score for each class.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature`)
3. Make changes and commit them (`git commit -am 'Add feature'`)
4. Push to the branch (`git push origin feature`)
5. Create a new pull request
