
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image 
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# Professional design
st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
}

[data-testid="stSidebar"] {
    background-color: #14213D;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #14213D;
    text-align: center;
}

.subtitle {
    font-size: 16px;
    color: #64748B;
    text-align: center;
    margin-bottom: 30px;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #E2E8F0;
    padding: 20px;
    border-radius: 12px;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# Load saved model and evaluation results
data = joblib.load("knn_digit_model.pkl")

knn = data["model"]
accuracy = data["accuracy"]
cm = data["confusion_matrix"]
report = data["classification_report"]


st.markdown(
    '<div class="main-title">🔢 Handwritten Digit Recognition using KNN</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">An Interactive Machine Learning Dashboard powered by the MNIST Dataset</div>',
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    ["Dashboard Overview", "Model Evaluation", "Digit Prediction"]
)

# PAGE 1: DASHBOARD OVERVIEW
if page == "Dashboard Overview":

    st.header("Project Overview")

    st.write("""
    This project uses the K-Nearest Neighbors algorithm
    to recognize handwritten digits from the MNIST dataset.
    """)

    st.subheader("Model Performance Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Training Images", "10,000")

    with col2:
        st.metric("Testing Images", "10,000")

    with col3:
        st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("Dataset Information")

    info1, info2, info3 = st.columns(3)

    with info1:
         st.markdown("**Dataset**")
         st.info("MNIST")

    with info2:
         st.markdown("**Image Dimensions**")
         st.info("28 × 28 Pixels")

    with info3:
         st.markdown("**Number of Classes**")
         st.info("10 Digits (0–9)")

    st.markdown("**Algorithm:** K-Nearest Neighbors (KNN)")
    st.markdown("**Number of Neighbors (K):** 5")

# PAGE 2: MODEL EVALUATION
elif page == "Model Evaluation":

    st.header("📊 Model Evaluation")

    st.write(
        "Evaluate the performance of the KNN model using "
        "accuracy, confusion matrix, and classification metrics."
    )

    st.divider()

    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    st.divider()
    st.subheader("📌 Confusion Matrix")
    st.caption(
          "The confusion matrix compares actual digits with the digits predicted by the KNN model."
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax
    )

    ax.set_xlabel("Predicted Digit")
    ax.set_ylabel("Actual Digit")
    ax.set_title("KNN Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("📋 Classification Report")

    st.caption(
        "This report displays the precision, recall, and F1-score "
        "for each digit from 0 to 9."
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
         report_df.round(3),
         use_container_width=True
    )

    st.divider()

    st.subheader("💡 Understanding the Evaluation Metrics")

    st.markdown("""
    - **Accuracy:** Measures the overall percentage of correct predictions.
    - **Precision:** Measures how many predicted digits were actually correct.
    - **Recall:** Measures how many actual digits were correctly identified.
    - **F1-score:** Combines precision and recall into a single score.
    - **Confusion Matrix:** Shows the correct and incorrect predictions for each digit.
    """)

# PAGE 3: DIGIT PREDICTION
elif page == "Digit Prediction":

    st.header("✍️ Handwritten Digit Prediction")

    st.write(
        "Test the trained KNN model by drawing your own handwritten digit."
    )

    st.info(
        "Instructions: Draw a single digit (0–9) in the canvas "
        "and click the Predict Digit button."
    )

    st.subheader("Draw Your Digit Below")

    canvas_col1, canvas_col2, canvas_col3 = st.columns([1, 2, 1])

    with canvas_col2:
        canvas_result = st_canvas(
             fill_color="#000000",
             stroke_width=18,
             stroke_color="#FFFFFF",
             background_color="#000000",
             height=280,
             width=280,
             drawing_mode="freedraw",
             return_image_data=True,
             key="digit_canvas"
        )

    if st.button("🔍 Predict Digit", type="primary", use_container_width=True):

        if canvas_result.image_data is not None:

            # Convert drawing to grayscale
            image = Image.fromarray(
                canvas_result.image_data.astype("uint8")
            ).convert("L")

            # Keep white strokes and remove background
            image = image.point(
                lambda p: 255 if p > 30 else 0
            )

            # Find digit boundaries
            bbox = image.getbbox()

            if bbox:

                # Crop digit
                image = image.crop(bbox)

                # Resize while maintaining proportions
                image.thumbnail(
                    (20, 20),
                    Image.Resampling.LANCZOS
                )

                # Create black 28x28 image
                final_image = Image.new("L", (28, 28), 0)

                # Center the digit
                x = (28 - image.width) // 2
                y = (28 - image.height) // 2

                final_image.paste(image, (x, y))

                # Normalize pixels
                image_array = np.array(
                    final_image
                ).astype("float32") / 255.0

                # Flatten image
                image_array = image_array.reshape(1, 784)

                # Predict using KNN
                prediction = knn.predict(image_array)

                predicted_digit = int(prediction[0])

                st.markdown("---")

                st.subheader("🎯 Prediction Result")

                st.success(
                    f"The recognized handwritten digit is: {predicted_digit}"
                )

                st.divider()

                st.subheader("🖼️ Processed Image")

                st.caption(
                     "The drawing is resized to 28 × 28 pixels and converted "
                     "into a format suitable for the KNN model."
                )

                st.image(final_image, width=200)

                # Display KNN voting score
                probabilities = knn.predict_proba(image_array)[0]
                st.subheader("📈 Prediction Confidence")

                confidence = probabilities[predicted_digit] * 100

                st.metric(
                     label="KNN Voting Score",
                     value=f"{confidence:.2f}%"
                )

                st.progress(float(confidence / 100))
                

            else:
                st.warning("Please draw a digit first.")

        else:
            st.warning("Please draw a digit first.")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Handwritten Digit Recognition"
)

st.sidebar.caption(
    "Machine Learning Project | KNN Algorithm"
)

st.sidebar.caption(
    "Built using Python and Streamlit"
)
