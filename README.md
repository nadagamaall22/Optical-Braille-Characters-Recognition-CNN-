# Optical Braille Character Recognition (CNN)

This repository contains a Jupyter Notebook for training and evaluating a machine learning model to classify Braille characters optically. The model is implemented using TensorFlow and Keras. The dataset, custom-made and preprocessed by the contributors, includes all 64 combinations of Braille characters represented in binary form (e.g., `000000` for no dots).

## Requirements

- Python 3.8 or higher
- Jupyter Notebook
- TensorFlow
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- OpenCV

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nadagamaall22/Optical-Braille-Characters-Recognition-CNN-.git
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

5. Open the `bcr-m.ipynb` notebook and run the cells to train and evaluate the model.

## Dataset

The dataset used in this project consists of 6519 images representing all 64 combinations of Braille characters in binary form. Each character is labeled with a six-bit binary code (e.g., `000000` for no dots). The dataset is preprocessed and split into training and testing sets within the notebook.

## Usage

1. Ensure all dependencies are installed as specified in the `requirements.txt`.
2. Open the Jupyter Notebook `bcr-m.ipynb` and follow the steps:
   - **Data Preprocessing**: Load and preprocess the custom Braille dataset.
   - **Model Building**: Define the neural network architecture using TensorFlow and Keras.
   - **Model Training**: Train the model on the training dataset.
   - **Model Evaluation**: Evaluate the model's performance on the test dataset.

## Results

The model's performance on the test set is evaluated in terms of accuracy and loss. The final results are printed in the notebook, demonstrating the effectiveness of the model in classifying Braille characters.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow
- Keras
- scikit-learn
- NumPy
- pandas
- matplotlib
- seaborn
- OpenCV

## Contributors
### Dataset Creation
- Adham Mohamed
- Ahmed Elsayed
- Nada Tarek
- Ali Elneklawi
- Ayman Feteha
- Abdallah Ashraf

### Model Development
- Nada Gamal El-Dien
