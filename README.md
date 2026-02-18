<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
</head>
<body>

<h1>🔥 Calories Burnt Prediction using Machine Learning</h1>

<div class="box">
    <p>
        This project focuses on predicting the number of 
        <span class="highlight">calories burned during exercise</span>
        using multiple Regression models.  
        The entire pipeline is built using <b>classes, objects, and modular functions</b> 
        to ensure clean architecture, scalability, and production-ready deployment.
    </p>
</div>

<h2>📌 Project Objective</h2>
<ul>
    <li>Analyze exercise and calorie datasets</li>
    <li>Merge and clean raw data</li>
    <li>Perform feature engineering and scaling</li>
    <li>Train multiple regression models</li>
    <li>Compare model performance using evaluation metrics</li>
    <li>Select the best-performing model for deployment</li>
</ul>

<h2>🧱 Project Architecture</h2>
<div class="diagram">
Exercise Data + Calories Data
   ➞
Data Merging
   ➞
Data Cleaning
   ➞
Train-Test Split
   ➞
Outlier Detection & Treatment
   ➞
Categorical Encoding
   ➞
Feature Scaling
   ➞
Model Training
   ➞
Model Evaluation
   ➞
Best Model Selection
   ➞
Pickle File
   ➞
Web Deployment
</div>

<h2>🧹 Data Preprocessing</h2>

<h3>1️⃣ Data Merging</h3>
<ul>
    <li>Merged <b>exercise.csv</b> and <b>calories.csv</b> using User_ID</li>
    <li>Removed unnecessary columns</li>
    <li>Verified dataset structure and integrity</li>
</ul>

<h3>2️⃣ Train-Test Split</h3>
<ul>
    <li>Applied 80-20 split using <b>train_test_split</b></li>
    <li>Used <b>random_state=42</b> for reproducibility</li>
</ul>

<h3>3️⃣ Outlier Detection & Handling</h3>
<ul>
    <li>Detected outliers using boxplots and statistical methods</li>
    <li>Applied <b>log transformation</b> and quantile capping</li>
    <li>Reduced skewness and improved model stability</li>
</ul>

<h3>4️⃣ Categorical Encoding</h3>
<ul>
    <li>Applied <b>One-Hot Encoding</b> on Gender feature</li>
    <li>Converted categorical values into numerical format</li>
</ul>

<h3>5️⃣ Feature Scaling</h3>
<ul>
    <li>Applied <b>StandardScaler</b></li>
    <li>Fitted on training data</li>
    <li>Transformed both train and test data</li>
    <li>Saved scaler for deployment</li>
</ul>

<h2>🤖 Machine Learning Models Used</h2>
<ul>
    <li>Linear Regression</li>
    <li>Ridge Regression</li>
    <li>Lasso Regression</li>
</ul>

<h2>📊 Model Evaluation</h2>
<div class="box">
    <ul>
        <li>Train-Test evaluation performed</li>
        <li>Each model trained on identical processed dataset</li>
        <li>Performance evaluated using:</li>
        <ul>
            <li>R² Score</li>
            <li>Mean Squared Error (MSE)</li>
            <li>Mean Absolute Error (MAE)</li>
        </ul>
    </ul>
</div>

<ul>
    <li>Compared R² scores across models</li>
    <li>Analyzed training vs testing performance</li>
    <li><b>Best-performing regression model</b> selected based on stability and accuracy</li>
</ul>

<h2>🏆 Best Model Selection</h2>
<div class="box">
    <p>
        Based on performance metrics and generalization ability,  
        the <span class="highlight">best regression model</span> was selected 
        and finalized for deployment.
    </p>
</div>

<h2>💾 Model Saving</h2>
<ul>
    <li>Saved trained model using <b>Pickle</b></li>
    <li>Saved scaler and feature configuration files</li>
    <li>Enabled reuse without retraining</li>
</ul>

<h2>🌐 Deployment</h2>
<ul>
    <li>Integrated trained model with a web application</li>
    <li>Connected Pickle files to backend</li>
    <li>Users can input exercise details and receive real-time calorie predictions</li>
</ul>

<h2>✨ Key Highlights</h2>
<ul>
    <li>End-to-End Machine Learning pipeline</li>
    <li>Object-Oriented Programming (OOP) implementation</li>
    <li>Robust preprocessing techniques</li>
    <li>Multiple regression model comparison</li>
    <li>Production-ready architecture</li>
</ul>

<h2>✅ Conclusion</h2>
<div class="box">
    <p>
        This project demonstrates a complete 
        <b>end-to-end Machine Learning regression solution</b> for 
        <span class="highlight">Calories Burnt Prediction</span>, 
        starting from raw dataset merging to final deployment 
        as a web application.
    </p>
</div>

</body>
</html>
