💳 Credit Card Fraud Analysis using Machine Learning
🚀 Overview

This project focuses on detecting fraudulent credit card transactions using machine learning algorithms and data analysis techniques. It leverages Django for the web interface, pandas and scikit-learn for model building, and Streamlit/Plotly for visualization. The system helps identify potentially fraudulent transactions with high accuracy and provides insights into fraud patterns through interactive dashboards.

🧠 Key Features

🧩 Machine Learning Models: Implements classification algorithms such as Logistic Regression, Random Forest, and XGBoost for fraud detection.

📊 Data Visualization: Interactive charts using Plotly, Seaborn, and Matplotlib for exploring transaction distributions and correlations.

💡 Model Explainability: Integrated with LIME and SHAP libraries for interpretable AI insights.

🌐 Django Web Interface: User-friendly dashboard for viewing analysis results and uploading transaction datasets.

⚙️ Automated Data Preprocessing: Handles missing data, outlier detection, and feature scaling seamlessly.

🛠️ API Integration (Optional): REST API endpoints using Django REST Framework for integrating fraud predictions into other systems.

🧰 Tech Stack
Category	Technologies Used
Frontend	HTML5, CSS3, Bootstrap, JavaScript
Backend	Django 4.1.1, Django REST Framework
Machine Learning	scikit-learn, XGBoost, LightGBM, imbalanced-learn
Visualization	Matplotlib, Seaborn, Plotly, Streamlit
Database	SQLite3 / MySQL
Others	Pandas, NumPy, Joblib, TensorFlow (optional for deep models)


⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/creditcard-fraud-analysis.git
cd creditcard-fraud-analysis

2️⃣ Create and Activate a Virtual Environment
python -m venv venv
venv\Scripts\activate      # On Windows
# or
source venv/bin/activate   # On macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Apply Database Migrations
python manage.py makemigrations
python manage.py migrate

5️⃣ Run the Server
python manage.py runserver


Then open your browser and visit:
👉 http://127.0.0.1:8000/

📊 Dataset

The dataset used for this project is the Credit Card Fraud Detection dataset from Kaggle, containing transactions made by European cardholders in September 2013.

Dataset link: Credit Card Fraud Detection (Kaggle)

Features: 30 (V1–V28 PCA components, Amount, Time, and Class)

Target: Class → 1 = Fraudulent, 0 = Normal

📈 Model Performance
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	98.7%	94%	92%	93%
Random Forest	99.3%	97%	96%	96%
XGBoost	99.5%	98%	98%	98%

(values may vary depending on dataset split and parameters)

🧮 Visualization Examples

Transaction Amount Distribution

Fraud vs Non-Fraud Pie Charts

Correlation Heatmaps

ROC and Precision-Recall Curves

🧑‍💻 Author

Tejasri Chandu

💼 Frontend Developer | Data Science Enthusiast

🌐 GitHub

✉️ Email: [your email]

📜 License

This project is licensed under the MIT License — feel free to use and modify it for learning or research purposes.
