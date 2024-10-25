<h1 align="center">Midterm: Linear Regression and Logistic Regression</h1>
<h2 align="center">MExEE 402 - MExE Elective 2</h2>
<br>

## Table of Contents
  - [I. Abstract](#i-abstract)
  - [II. Introduction](#ii-introduction)
  - [III. Dataset Description](#iii-dataset-description)
  - [IV. Project Objectives](#iv-project-objectives)
  - [V. Linear Regression Analysis](#v-linear-regression-analysis---bike-rental-dataset)
      - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
      - [Data Preprocessing](#data-preprocessing)
      - [Model Implementation](#model-implementation)
      - [Model Evaluation](#model-evaluation)
        - [Key Metrics](#key-metrics)
      - [Results](#results)
      - [Visuals](#visuals)
  - [VI. Logistic Regression Analysis](#vi-logistic-regression-analysis---banknote-authentication-dataset)
      - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
      - [Data Preprocessing](#data-preprocessing)
      - [Model Implementation](#model-implementation)
      - [Model Evaluation](#model-evaluation)
        - [Key Metrics](#key-metrics)
      - [Results](#results)
      - [Visuals](#visuals)
      - [Interpretation](#interpretation)
  - [VII. Documentation](#vii-documentation)
  - [VIII. Refererences](#viii-references)
  - [IX. Group Members](#ix-group-members)
<hr> 
<br>


## I. Abstract
<table>
  <tr>
    <td width="25%">
      <img src="https://github.com/user-attachments/assets/e6c3e5c0-a456-486d-92b8-8e894f92eab2" alt="Bike Sharing Dataset" style="width: 100%; height: 450px;">
    </td>
    <td width="50%">
      <div align="justify">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This <b>Pair-Based Project</b> showcases our understanding of <b>Linear</b> and <b>Logistic Regression</b> through hands-on analysis of two significant datasets: the <b>Bike Sharing Dataset</b> and the <b>Bank Note Authentication UCI data</b>. By documenting our methodologies, performance evaluations, and interpretations of the regression analyses, we aim to highlight the practical applications of these statistical techniques in real-world scenarios. <br> <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The <b>Bike Sharing Dataset</b> provides insights into bike rental patterns, influenced by environmental and temporal factors, allowing us to predict rental counts. In contrast, the <b>Bank Note Authentication Dataset</b> serves as a crucial resource for exploring classification techniques to detect counterfeit currency, enabling us to distinguish between authentic and forged banknotes based on numerical features. <br> <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This repository not only reflects our analytical capabilities but also our commitment to applying data-driven methodologies in solving practical problems.
      </div>
    </td>
    <td width="25%">
      <img src="https://github.com/user-attachments/assets/86b4a150-33f0-4fe8-b9b9-9c98f047efb7" alt="Investigator" style="width: 100%; height: 450px;">
    </td>
  </tr>
</table>

<br>


## II. Introduction

<p align="justify">  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This project is part of the <b>MeXE 402 - Mechatronics Engineering Elective 2: Data Science and Machine Learning</b> course midterm exam. The goal of this project is to analyze the real-world datasets <b>Bike Sharing Dataset</b> and <b>Bank Note Authentication UCI data</b> by applying: </p>

- **Linear Regression**
- **Logistic Regression**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; These two models are essential tools in data science for predicting and classifying outcomes based on input variables.
<br>

## Overview of Linear Regression

<p align="justify">  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Linear Regression</b> is a statistical method used to model the relationship between a dependent variable and one or more independent variables. The goal is to fit a straight line that best represents the relationship between the variables by minimizing the sum of squared differences between the observed values and the predicted values. It's commonly used for prediction and forecasting tasks, such as: </p>

- Predicting sales based on advertising expenditure.
- Estimating house prices from features like area, number of rooms, etc.


### Linear Regression Equation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The general equation for a linear regression model can be represented as:

**$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon
$$**

Where:
- <b>y</b> is the dependent variable (e.g., number of bike rentals).
- <b>X<sub>1</sub>, X<sub>2</sub>, ..., X<sub>n</sub></b> are the independent variables.
- <b>&beta;<sub>0</sub></b> is the intercept.
- <b>&beta;<sub>1</sub>, &beta;<sub>2</sub>, ..., &beta;<sub>n</sub></b> are the coefficients.
- <b>&epsilon;</b> is the error term.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In this project, we will apply Linear Regression to the **Bike Sharing Dataset** to predict the number of bike rentals based on weather conditions, season, and other factors.

<br>

## Overview of Logistic Regression

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Logistic Regression** is used when the dependent variable is binary (e.g., yes/no, true/false, or 0/1). It estimates the probability that a given input belongs to a specific category. The model outputs probabilities, which are then thresholded to classify the input into one of two classes.

### Logistic Regression Equation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For logistic regression, the model predicts the probability of a binary outcome, expressed as:

**$$
P(Y=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n)}}
$$**

Where:
- <b>P(Y=1 | X)</b> is the probability that the dependent variable <b>Y</b> equals 1 given the independent variables <b>X</b>.
- <b>e</b> is the base of the natural logarithm.
- <b>&beta;<sub>0</sub>, &beta;<sub>1</sub>, &beta;<sub>2</sub>, ..., &beta;<sub>n</sub></b> are the model coefficients.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In this project, we will apply Logistic Regression to the **Bank Note Authentication UCI Dataset** to classify banknotes as either authentic or counterfeit based on their features.

<br>



## III. Dataset Description 

### **1. Bike Sharing Dataset**

<img align="right" src="https://thumbs.dreamstime.com/b/citibike-bicycle-share-new-york-april-bicycles-citibank-s-program-await-riders-new-york-city-photo-taken-april-40609911.jpg" style="height: 330px;"></p>

- **Information:**
    <p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Bike sharing systems are modern versions of traditional rentals, automating membership, rental, and returns. Users can rent and return bikes at different locations, with over 500 programs and 500,000 bicycles worldwide. These systems are gaining attention for their role in addressing traffic, environmental, and health concerns. Unlike other transport services, bike sharing records precise data on travel duration, and departure/arrival points, effectively turning the system into a virtual sensor network that can help monitor city mobility and detect important events.
    </p>

- **Description:**
  <p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This dataset includes data on bike rentals across multiple seasons, with features capturing various environmental and temporal conditions. The dataset comprises several variables such as temperature, humidity, wind speed, season, holiday, working day, and weather situations. Each row represents daily rentals, which are influenced by these factors. The data provides an excellent opportunity to predict the number of rentals based on environmental factors. The main target variable is the number of rentals (count), which we aim to predict using <b>Linear Regression</b>.
    </p>


- **Attributes:**
<img align="right" src="https://jugnoo.io/wp-content/uploads/2021/08/bike.jpg" style="height: 200px;"></p>
  - **`season`**: season (1: spring, 2: summer, 3: fall, 4: winter)
  - **`yr`**: year (0: 2011, 1:2012)
  - **`mnth`**: month ( 1 to 12)
  - **`holiday`**: Whether the day is a holiday or not.
  - **`workingday`**: If day is neither weekend nor holiday is 1, otherwise is 0.
  - **`temp`**: Normalized temperature in Celcius.
  - **`hum`**: Normalized humidity. The values are divided to 100 (max)
  - **`windspeed`**: Normalized wind speed. The values are divided to 67 (max)
  - **`cnt`**: Count of total rental bikes including both casual and registered
<br>

### **Banknote Authentication Dataset**

<img align="right" src="https://st2.depositphotos.com/1538722/11652/i/450/depositphotos_116526898-stock-photo-euro-paper-banknotes.jpg" style="height: 320px;"></p>

- **Information:**
    <p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The Banknote Authentication Dataset is a valuable resource for researchers and data scientists interested in exploring classification techniques for counterfeit detection. With the increasing prevalence of forged currency, the ability to accurately distinguish between authentic and forged banknotes is crucial for financial institutions and security agencies. This dataset provides a unique opportunity to apply machine learning algorithms to a real-world problem, allowing practitioners to develop robust models that can effectively classify banknotes based on various numerical features derived from image analysis.
    </p>

- **Description:**
  <p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This dataset consists of features extracted from images of banknotes using wavelet transformation. The primary task is to classify the banknotes as <b>authentic</b> or <b>forged</b>. The dataset includes four numerical features: variance of wavelet-transformed image, skewness of wavelet-transformed image, kurtosis of wavelet-transformed image, and entropy of image. These features serve as the independent variables, while the target variable (class) is binary, representing whether the banknote is real or fake. <b>Logistic Regression</b> is used to perform binary classification.
    </p>

<img align="right" src="https://banknote-solutions.koenig-bauer.com/fileadmin/user_upload/News/Banknote/2022/22-025-L-Coverno-VariCash/KB_BNS_Digital_Solutions_ValiCash_Authenticate-1500x1000px.jpg" style="height: 180px;"></p>

- **Attributes:**
  - **`variance`**: Variance of wavelet-transformed image.
  - **`skewness`**: Skewness of wavelet-transformed image.
  - **`curtosis`**: Kurtosis of wavelet-transformed image.
  - **`entropy`**: Entropy of the image.
  - **`class`**: Binary class (0 = forged, 1 = authentic).
    
<br>



## IV. Project Objectives

The primary objectives of this project are:
- To develop a **Linear Regression** model to predict daily bike rentals using the **Bike Sharing Dataset**.
- To implement a **Logistic Regression** model to classify banknotes as authentic or forged using the **Banknote Authentication Dataset**.
- To document and interpret the results of each model, comparing the effectiveness and limitations of Linear and Logistic Regression.
- To improve our programming skills in Python, particularly with libraries like Scikit-learn and Pandas, through hands-on experience in building and evaluating machine learning models.
- To enhance our analytical skills by understanding how to analyze model outputs and make data-driven decisions based on our findings.

<br>


## V. **Linear Regression Analysis** - **Bike Rental Dataset**
<div align="justify">

  
### **Overview**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The project performs a **Linear Regression analysis** to **predict bike rental counts** based on **environmental** and **calendar-based factors**. The dataset includes features like **temperature**, **humidity**, **weather conditions**, and more. The analysis covers **data exploration**, **preprocessing**, **model training**, and **evaluation**.

### **Exploratory Data Analysis (EDA)**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The dataset consists of **731 observations** and **16 features**. Key variables include **temperature**, **humidity**, **windspeed**, and **total bike rental counts**. Columns are **renamed for clarity**, and the dataset is inspected for **missing values** and **data types**. Initial data exploration reveals **correlations** between **weather conditions** and **bike rental counts**.

### **Data Preprocessing**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Missing values** are handled using **K-Nearest Neighbors (KNN) imputation**, ensuring **data consistency**. **Outliers** and **extreme values** are checked to prevent **skewing the model results**. Continuous features are **normalized** to improve **model performance**. Features are split into **training** and **test sets** to evaluate the model's **generalizability**.

### **Model Implementation**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A **Linear Regression model** is applied using the **`Scikit-learn` library**. The model uses features such as **temperature**, **humidity**, and **windspeed** to predict **total bike rentals**. The training process involves **fitting the model** to the dataset and evaluating its **predictive power**.

### **Model Evaluation**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The model is evaluated using **R-squared**, **Mean Squared Error (MSE)**, and **Mean Absolute Error (MAE)**. These metrics provide insights into how well the model explains the **variability** in bike rentals and the **accuracy** of its predictions. The **coefficients** of the model are interpreted to understand the **impact of individual predictors**.

### **Key Metrics:**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- **R-squared:** Indicates the proportion of **variance** in bike rentals explained by the model. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- **MSE & MAE:** Measure the **average errors** in prediction, with lower values indicating **better performance**.

### **Results**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The analysis shows a **significant relationship** between **temperature** and **bike rentals**, with a **positive coefficient** suggesting **higher rentals** during **warmer conditions**. **Humidity** and **windspeed** also contribute to predicting bike rentals, although to a **lesser extent**.

### **Visuals:**

# Data Visualization Techniques

This section provides visualizations used in the analysis and modeling of the dataset.

| Description | Visualization |
|-------------|---------------|
| **Boxplots**: Used to detect and handle outliers. | <img src="https://github.com/user-attachments/assets/84177ace-26f5-4d5c-85dc-a7d9ef3f775a" alt="Boxplots" width="400"/> |
| **Normal Probability Plot**: Identifies substantive departures from normality and assesses goodness of fit. | <img src="https://github.com/user-attachments/assets/403246c3-b1ed-48cb-aa5c-a58916928aa7" alt="Normal Probability Plot" width="400"/> |
| **Distribution Plots**: Visualizations of key predictors and the target variable (**`total_count`**). | <img src="https://github.com/user-attachments/assets/61d931f3-dc27-4145-b196-cfc37793b052" alt="Distribution Plots" width="400"/> |
| **Heatmaps**: Visualizes correlations between variables. | <img src="https://github.com/user-attachments/assets/4778def5-84a2-41ea-923b-08f837c93989" alt="Heatmaps" width="400"/> |
| **Pair Plots**: Used for visualizing relationships among multiple predictors and their relationship with the target variable. | <img src="https://github.com/user-attachments/assets/6fe489b9-afbe-4ad6-9a76-aa92f12ee567" alt="Pair Plots" width="400"/> |
| **Cross-Validation Prediction Plot**: Shows the finite variance between actual and predicted target values. | <img src="https://github.com/user-attachments/assets/46a9a2c6-cb5e-45ac-9f02-0adfebed9759" alt="Cross-Validation Prediction Plot" width="400"/> |
| **Residual Plots**: Assesses model assumptions like linearity and homoscedasticity. | <img src="https://github.com/user-attachments/assets/99fb8d48-3e6b-4b2b-94f3-9aedbebc4e25" alt="Residual Plots" width="400"/> |
| **Scatter Plot**: Shows the relationship between actual and predicted values. | <img src="https://github.com/user-attachments/assets/124fd0b6-d70e-4d05-9f0d-58cb3ee27fb6" alt="Scatter Plot" width="400"/> |



</div>
<br>



## VI. **Logistic Regression Analysis** - **Banknote Authentication Dataset**

<div align="justify">
  
### **Overview**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This project applies **Logistic Regression** to classify **banknotes** as either **authentic** or **forged**, based on several statistical features derived from wavelet-transformed images of the banknotes. The dataset contains attributes such as **variance**, **skewness**, **curtosis**, and **entropy**. These features provide detailed information about the physical characteristics of each banknote, which the model uses to predict its authenticity. The project covers **data exploration**, **preprocessing steps**, **model training**, **evaluation**, and **interpretation**, culminating in a thorough analysis of the factors contributing to the model's classification ability.

### **Exploratory Data Analysis (EDA)**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The dataset consists of **1372 observations** and **5 key features**: **variance**, **skewness**, **curtosis**, **entropy**, and the target variable, **class**. Initial EDA is performed to understand the structure of the dataset, including checking for missing values and assessing the data types of each feature. The features are analyzed individually to observe their distribution, which is visualized using **histograms**. These plots reveal the spread and skewness of the data, showing how certain features, like variance, tend to have higher dispersion compared to others. **Outliers** are identified using **boxplots**, especially for features like skewness and curtosis, where extreme values are common.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The dataset is further explored using **pairplots**, which show the pairwise relationships between the features, color-coded by class. This helps visualize any linear or non-linear relationships between the variables and how they interact across the two classes (authentic vs. forged). A **correlation matrix** is visualized using a **heatmap**, which highlights the relationships between features such as variance and skewness. These insights lay the groundwork for selecting features that will have the most impact on the Logistic Regression model.

### **Data Preprocessing**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Missing values are checked using methods such as **K-Nearest Neighbors (KNN)**, but in this case, all features are fully populated, allowing us to proceed directly with analysis. Before training the model, several preprocessing steps are applied to ensure optimal performance. Outliers are imputed, and the data is scaled using the **StandardScaler** to standardize features like **variance**, **skewness**, **curtosis**, and **entropy**. This scaling process ensures that all features have the same range, preventing any single feature from disproportionately influencing the model. The dataset is then split into a **training set** and a **test set**, allowing the model to learn from one portion of the data while being evaluated on unseen data. This ensures that the model's performance is generalized and not simply overfitting to the training data.

### **Model Implementation**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The **Logistic Regression model** is built using the **Scikit-learn** library, one of the most widely used machine learning frameworks. The model is trained on the four primary features: variance, skewness, curtosis, and entropy, with the target being the class label that distinguishes authentic banknotes from forgeries. The training process involves fitting the model to the training data, allowing it to learn the relationships between the input features and the output class.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Once the model is trained, it is used to make predictions on both the training and test sets. Additionally, the model is tested by predicting the class of a single banknote data point to demonstrate its practical application.

### **Model Evaluation**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The performance of the **Logistic Regression model** is evaluated using several **key metrics**. The **confusion matrix** is generated, which provides detailed information on the number of true positives, true negatives, false positives, and false negatives. This matrix allows us to assess the model's ability to correctly classify **authentic** and **forged banknotes**. **Accuracy**, which is a key metric in binary classification, is calculated to determine the overall success rate of the model. The **coefficients** of the Logistic Regression model are also analyzed to understand how each feature contributes to the final classification. For example, a positive coefficient for variance suggests that as variance increases, the likelihood of the banknote being classified as authentic increases. These insights are valuable for interpreting the model's decision-making process.

### **Key Metrics**
- **Accuracy**: The overall accuracy of the model, which represents the proportion of correctly classified instances.
- **Confusion Matrix**: This provides a breakdown of correct and incorrect classifications, helping to identify the model's strengths and areas for improvement.
- **Model Coefficients**: The coefficients associated with variance, skewness, curtosis, and entropy indicate the importance of these features in predicting whether a banknote is authentic.

### **Results**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The **Logistic Regression model** performs well, achieving **high accuracy** in classifying banknotes as either **authentic** or **forged**. Variance and curtosis emerge as the most significant predictors of banknote authenticity, with entropy having a smaller, but still relevant, impact. The confusion matrix shows that the model is particularly strong in avoiding false positives, ensuring that genuine banknotes are rarely misclassified as forgeries.

### **Visuals:**
- **Boxplots**: Used to detect and handle outliers.
- **Histograms**: Display the distribution of each feature.
- **Pairplots**: Visualize pairwise relationships between features and classes.
- **Heatmap**: Correlation matrix to show relationships among features.
- **Confusion Matrix**: Visualizes classification performance.
- **Bar Plot**: Displays feature importance based on the Logistic Regression model coefficients.

### Interpretation
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The analysis reveals that the Logistic Regression model can accurately distinguish between authentic and forged banknotes. The most important features for classification are variance and curtosis, which align with the physical characteristics of the banknotes. The model's high accuracy and detailed evaluation metrics suggest it is highly effective for this task. By analyzing the model coefficients, we can further understand how specific features contribute to the classification decisions, offering transparency into the model’s decision-making process.

</div>
<br>



## VII. Documentation
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Discussion: Reflect on the results, compare the two regression methods, and mention any limitations.
  </p>
<br>
<br>


## VIII. References

https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset <br>
https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data <br>
https://github.com/MikkoDT/MeXE402_Midterm_4102

<br>
<br>


## IX. Group Members

<div align="center">

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/2d9ebaa0-d550-4b60-856d-d2c98fb9f3d1" alt="Malata" style="height: 230px; float: left;"></td>
    <td><img src="https://github.com/yannaaa23/CSE-Testing-Feb-19/blob/6ef6454fddba5503da2057bcf06fe77ca1491e0c/IMG_20230605_215028_860.jpg" alt="Umali" style="height: 230px; float: left;"></td>
  </tr>
  <tr>
    <td align="center"><strong>Malata, John Rei R.</strong></td>
    <td align="center"><strong>Umali, Ariane Mae D.</strong></td>
  </tr>
</table>

</div>

<br>
<br>





