# 🌱 Eco-Friendly Product Recommendation App

## 🔍 Overview

This Streamlit application provides eco-friendly product recommendations using a user-friendly interface. The UI is designed with responsiveness in mind and supports a clean and modern aesthetic.

## 🛠 Technologies Used

- **🐍 Python** (for backend logic)
- **🎨 Streamlit** (for frontend UI and interactivity)
- **📊 Pandas** (for data handling)
- **💾 Joblib** (for loading models if needed)
- **🤖 Scikit-learn** (for machine learning models)
- **🎨 Custom CSS** (for enhanced styling)

## 🧠 Algorithm Used

This app employs the K-Nearest Neighbors (KNN) algorithm. It works as follows:
1️⃣ Finds the K most similar products based on features like eco-friendliness.
2️⃣ Measures similarity using distance calculations (like Euclidean distance).
3️⃣ Recommends products that are closest to the selected preferences.

## 🚀 Live Demo

👉[View Application](https://eco-friendly.streamlit.app/)

## ⚙️ Installation

1. Clone this repository:

   ```bash

   git clone https://github.com/simranq/eco-friendly-product-recommender.git
   cd eco-friendly-product-recommender
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## 🚀 Usage

- Launch the app and explore recommended products.
- The interface dynamically adjusts product cards for better readability on different screen sizes.
- Click on recommendations for more details.

## 📜 License

This project is licensed under the GNU AGPLv3 License.
