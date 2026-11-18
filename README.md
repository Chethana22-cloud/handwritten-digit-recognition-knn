# 🔢 Handwritten Digit Recognition using KNN
https://handwritten-digit-recognition-knn-tdmoec4syro7exbq36mbui.streamlit.app/
## 📌 Project Overview

This project focuses on recognizing handwritten digits (0–9) using the K-Nearest Neighbors (KNN) machine learning algorithm.

The project uses the MNIST dataset to train and evaluate the model. An interactive web application is developed using Streamlit, where users can draw a handwritten digit and get its predicted output.

## 🎯 Objectives

- To implement the K-Nearest Neighbors (KNN) classification algorithm.
- To use the MNIST dataset for handwritten digit recognition.
- To perform image preprocessing and normalization.
- To evaluate model performance using classification metrics.
- To develop an interactive web application using Streamlit.

## 📊 Dataset Information

- **Dataset:** MNIST Handwritten Digit Dataset
- **Source:** Keras
- **Training Images Used:** 10,000
- **Testing Images Used:** 10,000
- **Image Dimensions:** 28 × 28 pixels
- **Number of Classes:** 10 (Digits 0–9)

## ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Pillow

## 🤖 Machine Learning Algorithm

**K-Nearest Neighbors (KNN)**

KNN is a supervised machine learning algorithm used for classification. It predicts the class of a new input by identifying the nearest data points in the training dataset.

**Model Configuration:**
- Algorithm: K-Nearest Neighbors
- Number of Neighbors (K): 5
- Input Features: 784 pixels
- Pixel Values: Normalized between 0 and 1

## 📈 Model Performance

**Test Accuracy: 94.42%**

The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## 🌐 Application Features

The Streamlit application contains three main sections:

1. **Dashboard Overview**
   - Displays project information.
   - Shows training and testing image counts.
   - Displays model accuracy.

2. **Model Evaluation**
   - Displays the confusion matrix.
   - Shows the classification report.
   - Explains evaluation metrics.

3. **Digit Prediction**
   - Provides a drawing canvas.
   - Allows users to draw a single digit from 0 to 9.
   - Predicts the handwritten digit using the trained KNN model.
   - Displays the processed 28 × 28 image and KNN voting score.

## 📁 Project Structure

```text
handwritten-digit-recognition-knn/
│
├── app.py
├── knn_digit_model.pkl
├── requirements.txt
└── README.md
```

## 🚀 How to Run the Project Locally

**Step 1: Clone the repository**

```bash
git clone https://github.com/Chethana22-cloud/handwritten-digit-recognition-knn.git
```

**Step 2: Open the project folder**

```bash
cd handwritten-digit-recognition-knn
```

**Step 3: Install the required libraries**

```bash
pip install -r requirements.txt
```

**Step 4: Run the Streamlit application**

```bash
streamlit run app.py
```

**Step 5:** Open the local URL displayed in the terminal.
https://handwritten-digit-recognition-knn-tdmoec4syro7exbq36mbui.streamlit.app/

## 🔮 Future Enhancements

- Improve prediction accuracy through further experimentation.
- Explore other machine learning algorithms.
- Improve image preprocessing for handwritten inputs.
- Extend the application to recognize multi-digit numbers.

## 📝 Conclusion

This project demonstrates the application of the K-Nearest Neighbors algorithm for handwritten digit recognition using the MNIST dataset. It also demonstrates how a trained machine learning model can be integrated with an interactive Streamlit web application.

## 👩‍💻 Author

**Chethana C. B.**
