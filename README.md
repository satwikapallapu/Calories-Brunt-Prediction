<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Calories Burn Prediction System</title>
</head>
<body>

<div align="center">

<h1>
🔥 <span style="color:#ff4b2b;">Calories Burn Prediction System</span>
</h1>

<h3>
End-to-End Machine Learning Pipeline for Regression Modeling
</h3>

<p>
<b>✔ Modular Architecture</b> &nbsp; | &nbsp;
<b>✔ Outlier Treatment</b> &nbsp; | &nbsp;
<b>✔ Regularized Models</b> &nbsp; | &nbsp;
<b>✔ Deployment Ready</b>
</p>

</div>

<hr>

<h2>📌 Overview</h2>

<p>
A production-structured machine learning pipeline that predicts 
<strong>calories burned</strong> using physiological and exercise data.
</p>

<ul>
    <li>Proper Train/Test separation</li>
    <li>Log transformation & quantile capping</li>
    <li>Categorical encoding</li>
    <li>Feature scaling</li>
    <li>Linear, Ridge & Lasso regression</li>
    <li>Model serialization for deployment</li>
</ul>

<hr>

<h2>🏗 System Architecture</h2>

<pre>
Data Loading
     ↓
Dataset Merge
     ↓
Train-Test Split
     ↓
Outlier Treatment + Log Transformation
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Model Training (LR / Ridge / Lasso)
     ↓
Evaluation
     ↓
Model & Scaler Serialization
</pre>

<hr>

<h2>📂 Project Structure</h2>

<pre>
calories_prediction/
│
├── main.py
├── var_out.py
├── feature_selection.py
├── balanced_data.py
├── all_models.py
├── log_code.py
│
├── exercise.csv
├── calories.csv
│
├── plots_path/
│
├── scaler.pkl
├── calories.pkl
└── feature_selection.pkl
</pre>

<hr>

<h2>🤖 Models Implemented</h2>

<ul>
    <li><strong>Linear Regression</strong> – Baseline Model</li>
    <li><strong>Ridge Regression</strong> – L2 Regularization</li>
    <li><strong>Lasso Regression</strong> – L1 Regularization</li>
</ul>

<hr>

<h2>📈 Evaluation Metrics</h2>

<ul>
    <li>R² Score</li>
    <li>Mean Squared Error (MSE)</li>
    <li>Mean Absolute Error (MAE)</li>
</ul>

<hr>

<h2>🚀 How to Run</h2>

<pre>
pip install -r requirements.txt
python main.py
</pre>

<hr>

<h2>🎯 Design Principles</h2>

<ul>
    <li>Reproducibility</li>
    <li>Data leakage prevention</li>
    <li>Modular pipeline design</li>
    <li>Deployment readiness</li>
</ul>

<hr>

<p align="center"><b>Built with Python & Scikit-Learn</b></p>

</body>
</html>
