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
      - [Interpretation](#interpretation)
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
      - [Methodology](#methodology)
         - [Linear Regression](#linear-regression)
         - [Logistic Regression](#logistic-regression)
      - [Results Reflection](#results-reflection)
      - [Comparison of Regression Methods](#comparison-of-regression-methods)
      - [Limitations](#limitations)
  - [VIII. Refererences](#viii-references)
  - [IX. Group Members](#ix-group-members)
<hr> 
<br>


## I. Abstract
<table>
  <tr>
    <td width="27%">
      <img src="https://github.com/user-attachments/assets/e6c3e5c0-a456-486d-92b8-8e894f92eab2" alt="Bike Sharing Dataset" style="width: 100%; height: 450px;">
    </td>
    <td width="46%">
      <div align="justify">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This <b>Pair-Based Project</b> showcases our understanding of <b>Linear</b> and <b>Logistic Regression</b> through hands-on analysis of two significant datasets: the <b>Bike Sharing Dataset</b> and the <b>Bank Note Authentication UCI data</b>. By documenting our methodologies, performance evaluations, and interpretations of the regression analyses, we aim to highlight the practical applications of these statistical techniques in real-world scenarios. <br> <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The <b>Bike Sharing Dataset</b> provides insights into bike rental patterns, influenced by environmental and temporal factors, allowing us to predict rental counts. In contrast, the <b>Bank Note Authentication Dataset</b> serves as a crucial resource for exploring classification techniques to detect counterfeit currency, enabling us to distinguish between authentic and forged banknotes based on numerical features. <br> <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This repository not only reflects our analytical capabilities but also our commitment to applying data-driven methodologies in solving practical problems.
      </div>
    </td>
    <td width="27%">
      <img src="https://github.com/user-attachments/assets/86b4a150-33f0-4fe8-b9b9-9c98f047efb7" alt="Investigator" style="width: 100%; height: 450px;">
    </td>
  </tr>
</table>

<br>


## II. Introduction
- This project is part of the <b>MeXE 402 - Mechatronics Engineering Elective 2: Data Science and Machine Learning</b> course midterm exam.
- The goal of this project is to analyze the real-world datasets <b>Bike Sharing Dataset</b> and <b>Bank Note Authentication UCI data</b> by applying:
  - **Linear Regression**
  - **Logistic Regression**
- These two models are essential tools in data science for predicting and classifying outcomes based on input variables.

<br>

## Overview of Linear Regression
  - <b>Linear Regression</b> is a statistical method used to model the relationship between a dependent variable and one or more independent variables.
  - The goal is to fit a straight line that best represents the relationship between the variables by minimizing the sum of squared differences between the observed values and the predicted values.
  - It's commonly used for prediction and forecasting tasks, such as: </p>
    - Predicting sales based on advertising expenditure.
    - Estimating house prices from features like area, number of rooms, etc.


### Linear Regression Equation
  - The general equation for a linear regression model can be represented as:

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
- **Logistic Regression** is used when the dependent variable is binary (e.g., yes/no, true/false, or 0/1).
- It estimates the probability that a given input belongs to a specific category.
- The model outputs probabilities, which are then thresholded to classify the input into one of two classes.

### Logistic Regression Equation
- For logistic regression, the model predicts the probability of a binary outcome, expressed as:

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
  - Bike sharing systems are modern versions of traditional rentals, automating membership, rental, and returns.
  - Users can rent and return bikes at different locations, with over 500 programs and 500,000 bicycles worldwide. These systems are gaining attention for their role in addressing traffic, environmental, and health concerns.
  - Unlike other transport services, bike sharing records precise data on travel duration, and departure/arrival points, effectively turning the system into a virtual sensor network that can help monitor city mobility and detect important events.
    </p>

- **Description:**
  - This dataset includes data on bike rentals across multiple seasons, with features capturing various environmental and temporal conditions.
  - The dataset comprises several variables such as temperature, humidity, wind speed, season, holiday, working day, and weather situations.
  - Each row represents daily rentals, which are influenced by these factors.
  - The data provides an excellent opportunity to predict the number of rentals based on environmental factors. The main target variable is the number of rentals (count), which we aim to predict using <b>Linear Regression</b>.
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

### **2. Banknote Authentication Dataset**

<img align="right" src="https://st2.depositphotos.com/1538722/11652/i/450/depositphotos_116526898-stock-photo-euro-paper-banknotes.jpg" style="height: 320px;"></p>

- **Information:**
  - The Banknote Authentication Dataset is a valuable resource for researchers and data scientists interested in exploring classification techniques for counterfeit detection.
  - With the increasing prevalence of forged currency, the ability to accurately distinguish between authentic and forged banknotes is crucial for financial institutions and security agencies.
  - This dataset provides a unique opportunity to apply machine learning algorithms to a real-world problem, allowing practitioners to develop robust models that can effectively classify banknotes based on various numerical features derived from image analysis.
    </p>

- **Description:**
  - This dataset consists of features extracted from images of banknotes using wavelet transformation.
  - The primary task is to classify the banknotes as <b>authentic</b> or <b>forged</b>.
  - The dataset includes four numerical features: variance of wavelet-transformed image, skewness of wavelet-transformed image, kurtosis of wavelet-transformed image, and entropy of image. These features serve as the independent variables, while the target variable (class) is binary, representing whether the banknote is real or fake. <b>Logistic Regression</b> is used to perform binary classification.
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

  
- ### **Overview**
  - The project performs a **Linear Regression analysis** to **predict bike rental counts** based on **environmental** and **calendar-based factors**.
  - Dataset includes features like **temperature**, **humidity**, **weather conditions**, and more.
  - The analysis covers **data exploration**, **preprocessing**, **model training**, and **evaluation**.

- ### **Exploratory Data Analysis (EDA)**
  - The dataset consists of **731 observations** and **16 features**.
  - Key variables include **temperature**, **humidity**, **windspeed**, and **total bike rental counts**. Columns are **renamed for clarity**, and the dataset is inspected for **missing values** and **data types**.
  - Initial data exploration reveals **correlations** between **weather conditions** and **bike rental counts**.

- ### **Data Preprocessing**
  -  **Missing values** are handled using **K-Nearest Neighbors (KNN) imputation**, ensuring **data consistency**.
  -  **Outliers** and **extreme values** are checked to prevent **skewing the model results**. Continuous features are **normalized** to improve **model performance**.
  -  Features are split into **training** and **test sets** to evaluate the model's **generalizability**.

- ### **Model Implementation**
  - A **Linear Regression model** is applied using the **`Scikit-learn` library**.
  - The model uses features such as **temperature**, **humidity**, and **windspeed** to predict **total bike rentals**.
  - The training process involves **fitting the model** to the dataset and evaluating its **predictive power**.

- ### **Model Evaluation**
  - The model is evaluated using **R-squared**, **Mean Squared Error (MSE)**, and **Mean Absolute Error (MAE)**. These metrics provide insights into how well the model explains the **variability** in bike rentals and the **accuracy** of its predictions.
  - The **coefficients** of the model are interpreted to understand the **impact of individual predictors**.

- ### **Key Metrics:**
  - **R-squared:** Indicates the proportion of **variance** in bike rentals explained by the model.
  - **MSE & MAE:** Measure the **average errors** in prediction, with lower values indicating **better performance**.

- ### **Results**
  - The analysis shows a **significant relationship** between **temperature** and **bike rentals**, with a **positive coefficient** suggesting **higher rentals** during **warmer conditions**.
  - **Humidity** and **windspeed** also contribute to predicting bike rentals, although to a **lesser extent**.

- ### **Visuals:**

  - ### Data Visualization Techniques

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This section provides visualizations used in the analysis and modeling of the dataset.

| Description | Visualization |
|-------------|---------------|
| **Boxplots**: Used to detect and handle outliers. | <img src="https://github.com/user-attachments/assets/84177ace-26f5-4d5c-85dc-a7d9ef3f775a" alt="Boxplots" width="400"/> |
| **Normal Probability Plot**: Identifies substantive departures from normality and assesses goodness of fit. | <img src="https://github.com/user-attachments/assets/f6d97bb6-a65c-4ba4-9727-792523fbda05" alt="Normal Probability Plot" width="400"/> |
| **Distribution Plots**: Visualizations of key predictors and the target variable (**`total_count`**). | <img src="https://github.com/user-attachments/assets/647e5b16-d1bd-4b7e-8075-33a641e618ee" alt="Distribution Plots" width="400"/> |
| **Heatmaps**: Visualizes correlations between variables. | <img src="https://github.com/user-attachments/assets/01f1b913-7d9e-4f4c-9fd5-80782c466af5" alt="Heatmaps" width="400"/> |
| **Pair Plots**: Used for visualizing relationships among multiple predictors and their relationship with the target variable. | <img src="https://github.com/user-attachments/assets/718a1d23-78bd-404f-bde1-5e29330e1e3e" alt="Pair Plots" width="400"/> |
| **Cross-Validation Prediction Plot**: Shows the finite variance between actual and predicted target values. | <img src="https://github.com/user-attachments/assets/c32767bd-2285-44dc-9b60-65e6133a5c39" alt="Cross-Validation Prediction Plot" width="400"/> |
| **Residual Plots**: Assesses model assumptions like linearity and homoscedasticity. | <img src="https://github.com/user-attachments/assets/ff159cf3-e9dd-4916-ab9e-37d3eda73ec5" alt="Residual Plots" width="400"/> |
| **Scatter Plot**: Shows the relationship between actual and predicted values. | <img src="https://github.com/user-attachments/assets/b90baa8c-9145-4584-b150-18404fae6a38" alt="Scatter Plot" width="400"/> |


- ### Interpretation

- **Model Coefficients**:
  - Each coefficient indicates the expected change in bike rentals per unit increase in that feature, with all others held constant.
  - **Positive coefficients** (e.g., for temperature) suggest an increase in rentals with higher values.
  - **Negative coefficients** (e.g., for humidity) imply a decrease in rentals as values increase.
  - The magnitude of each coefficient shows the strength of its impact on rental counts.

- **Predictive Power**:
  - **Accuracy Score**: The model achieved an accuracy score (computed on training data), giving an initial measure of how well it captures data patterns.
  - **R² Score**: Averaged across cross-validation, this score shows the proportion of rental count variability explained by the model’s features.
  - **Mean Absolute Error (MAE)**: Represents the average absolute error between predicted and actual rental counts, with lower values indicating closer predictions.

These results suggest that the model effectively leverages key features to forecast bike rentals accurately, while coefficients provide insights into the most influential factors.

</div>
<br>



## VI. **Logistic Regression Analysis** - **Banknote Authentication Dataset**

<div align="justify">
  
- ### **Overview**
  - This project applies **Logistic Regression** to classify **banknotes** as either **authentic** or **forged**, based on several statistical features derived from wavelet-transformed images of the banknotes.
  - The dataset contains attributes such as **variance**, **skewness**, **curtosis**, and **entropy**.
  - These features provide detailed information about the physical characteristics of each banknote, which the model uses to predict its authenticity.
  - The project covers **data exploration**, **preprocessing steps**, **model training**, **evaluation**, and **interpretation**, culminating in a thorough analysis of the factors contributing to the model's classification ability.

- ### **Exploratory Data Analysis (EDA)**
  - The dataset consists of **1372 observations** and **5 key features**: **variance**, **skewness**, **curtosis**, **entropy**, and the target variable, **class**.
  - Initial EDA is performed to understand the structure of the dataset, including checking for missing values and assessing the data types of each feature.
  - The features are analyzed individually to observe their distribution, which is visualized using **histograms**. These plots reveal the spread and skewness of the data, showing how certain features, like variance, tend to have higher dispersion compared to others.
  - **Outliers** are identified using **boxplots**, especially for features like skewness and curtosis, where extreme values are common.
  - The dataset is further explored using **pairplots**, which show the pairwise relationships between the features, color-coded by class. This helps visualize any linear or non-linear relationships between the variables and how they interact across the two classes (authentic vs. forged).
  - A **correlation matrix** is visualized using a **heatmap**, which highlights the relationships between features such as variance and skewness. These insights lay the groundwork for selecting features that will have the most impact on the Logistic Regression model.

- ### **Data Preprocessing**
  - Missing values are checked using methods such as **K-Nearest Neighbors (KNN)**, but in this case, all features are fully populated, allowing us to proceed directly with analysis.
  - Before training the model, several preprocessing steps are applied to ensure optimal performance. Outliers are imputed, and the data is scaled using the **StandardScaler** to standardize features like **variance**, **skewness**, **curtosis**, and **entropy**. This scaling process ensures that all features have the same range, preventing any single feature from disproportionately influencing the model.
  - The dataset is then split into a **training set** and a **test set**, allowing the model to learn from one portion of the data while being evaluated on unseen data. This ensures that the model's performance is generalized and not simply overfitting to the training data.

- ### **Model Implementation**
  - The **Logistic Regression model** is built using the **Scikit-learn** library, one of the most widely used machine learning frameworks.
  - The model is trained on the four primary features: variance, skewness, curtosis, and entropy, with the target being the class label that distinguishes authentic banknotes from forgeries.
  - The training process involves fitting the model to the training data, allowing it to learn the relationships between the input features and the output class.
  - Once the model is trained, it is used to make predictions on both the training and test sets.
  - Additionally, the model is tested by predicting the class of a single banknote data point to demonstrate its practical application.

- ### **Model Evaluation**
  - The performance of the **Logistic Regression model** is evaluated using several **key metrics**. The **confusion matrix** is generated, which provides detailed information on the number of true positives, true negatives, false positives, and false negatives. This matrix allows us to assess the model's ability to correctly classify **authentic** and **forged banknotes**.
  - **Accuracy**, which is a key metric in binary classification, is calculated to determine the overall success rate of the model.
  - The **coefficients** of the Logistic Regression model are also analyzed to understand how each feature contributes to the final classification. For example, a positive coefficient for variance suggests that as variance increases, the likelihood of the banknote being classified as authentic increases. These insights are valuable for interpreting the model's decision-making process.

- ### **Key Metrics**
  - **Accuracy**: The overall accuracy of the model, which represents the proportion of correctly classified instances.
  - **Confusion Matrix**: This provides a breakdown of correct and incorrect classifications, helping to identify the model's strengths and areas for improvement.
  - **Model Coefficients**: The coefficients associated with variance, skewness, curtosis, and entropy indicate the importance of these features in predicting whether a banknote is authentic.

- ### **Results**
  - The **Logistic Regression model** performs well, achieving **high accuracy** in classifying banknotes as either **authentic** or **forged**.
  - Variance and curtosis emerge as the most significant predictors of banknote authenticity, with entropy having a smaller, but still relevant, impact.
  - The confusion matrix shows that the model is particularly strong in avoiding false positives, ensuring that genuine banknotes are rarely misclassified as forgeries.



- ### **Visuals:**

  - ### Data Visualization Techniques

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This section presents visualizations used for analyzing and modeling the dataset.

| Description | Visualization |
|-------------|---------------|
| **Boxplots**: Used to detect and handle outliers. | <img src="https://github.com/yannaaa23/CSE-Testing-Feb-19/blob/2b02c2f707035cc126241f5b13b5fd99631679ea/Elec%20Mid/box%20plot.png" alt="Boxplots" width="1000"/> |
| **Pairplots**: Visualize pairwise relationships between features and classes. | <img src="https://github.com/yannaaa23/CSE-Testing-Feb-19/blob/2b02c2f707035cc126241f5b13b5fd99631679ea/Elec%20Mid/PIC%20PLOT/paiplot.png" alt="Pairplots" width="1000"/> |
| **Normal Probability Plot**: A graphical tool used to assess whether a dataset follows a normal distribution by plotting the data's quantiles against theoretical quantiles of a normal distribution, ideally forming a straight line if normally distributed. | <img src="https://github.com/yannaaa23/CSE-Testing-Feb-19/blob/2b02c2f707035cc126241f5b13b5fd99631679ea/Elec%20Mid/PIC%20PLOT/norm%20prob.png" alt="Normal Probability Plot" width="1000"/> |
| **Histograms**: Display the distribution of each feature. | <img src="https://github.com/yannaaa23/CSE-Testing-Feb-19/blob/2b02c2f707035cc126241f5b13b5fd99631679ea/Elec%20Mid/PIC%20PLOT/histplot.png" alt="Histograms" width="1000"/> |
| **Heatmap**: Correlation matrix to show relationships among features. | <img src="https://github.com/yannaaa23/CSE-Testing-Feb-19/blob/2b02c2f707035cc126241f5b13b5fd99631679ea/Elec%20Mid/PIC%20PLOT/heatmap.png" alt="Heatmap" width="1000"/> |
| **Bar Plot**: Displays feature importance based on the Logistic Regression model coefficients. | <img src="https://github.com/yannaaa23/CSE-Testing-Feb-19/blob/2b02c2f707035cc126241f5b13b5fd99631679ea/Elec%20Mid/PIC%20PLOT/bar%20plot.png" alt="Bar Plot" width="1000"/> |
| **Confusion Matrix**: Visualizes classification performance. | <img src="https://github.com/yannaaa23/CSE-Testing-Feb-19/blob/2b02c2f707035cc126241f5b13b5fd99631679ea/Elec%20Mid/PIC%20PLOT/conf%20Mat.png" alt="Confusion Matrix" width="1000"/> |



- ### Interpretation
  - The analysis reveals that the Logistic Regression model can accurately distinguish between authentic and forged banknotes.
  - The most important features for classification are variance and curtosis, which align with the physical characteristics of the banknotes.
  - The model's high accuracy and detailed evaluation metrics suggest it is highly effective for this task.
  - By analyzing the model coefficients, we can further understand how specific features contribute to the classification decisions, offering transparency into the model’s decision-making process.

</div>
<br>



## VII. Documentation
<p align="justify"> 
  
### Methodology

#### Linear Regression

This analysis follows a structured approach, documented in the following steps:

1. **Exploratory Data Analysis (EDA)**
   - **Import Libraries**: Imported essential libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` for data handling, visualization, and modeling.
   - **Data Loading**: Loaded the dataset from a CSV file for analysis.
   - **Data Overview**: Examined the dataset shape, data types, and initial rows to understand its structure and characteristics.
   - **Renaming Columns**: Renamed columns to more descriptive names to improve readability.
   - **Typecasting**: Converted appropriate numerical columns to categorical data types to better represent the data.

2. **Data Preprocessing**
   - **Statistical Summary**: Generated summary statistics for numerical columns to identify key data properties.
   - **Handling Missing Values**: Addressed any missing values in the dataset using appropriate imputation methods (e.g., KNN).
   - **Splitting Data**: Divided the dataset into training and testing sets for model evaluation.

3. **Feature Engineering**
   - **Feature Selection and Transformation**: Selected relevant features and made transformations to enhance model performance.
   - **Categorical Encoding**: Converted categorical variables into numerical formats suitable for modeling.

4. **Model Building**
   - **Linear Regression Model**: Trained a linear regression model using the selected features.
   - **Evaluation Metrics**: Computed metrics such as R², Mean Absolute Error (MAE), and accuracy score to assess model performance.

5. **Model Interpretation and Analysis**
   - **Coefficients Analysis**: Interpreted model coefficients to understand the influence of each feature on bike rentals.
   - **Prediction Testing**: Used test data to evaluate the model’s predictive power and consistency.

This structured methodology ensures thorough data analysis and a systematic approach to building and evaluating the linear regression model.

#### Logistic Regression

This analysis follows a structured approach, documented in the following steps:

1. **Exploratory Data Analysis (EDA)**
    - **Import Libraries**: Imported essential libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn for data handling, visualization, and modeling.
      ```python
      from sklearn.preprocessing import StandardScaler
      import pandas as pd
      from sklearn.model_selection import train_test_split
      from sklearn.linear_model import LogisticRegression
      from sklearn.metrics import confusion_matrix, accuracy_score
      import matplotlib.pyplot as plt
      import seaborn as sns
      import numpy as np
      from sklearn.impute import KNNImputer
      ```
    This step involves importing essential libraries, including tools for data handling, model training, and performance evaluation.

    - **Data Loading**: Loaded the Banknote Authentication dataset from a CSV file for analysis.
      ```python
      banknote = pd.read_csv('banknote_authentication.csv')
      ```
      The dataset is loaded into a DataFrame named <b><code>banknote</code></b> for exploration and processing.
      
    - **Data Overview**: Examined the dataset shape, data types, and missing values to understand its structure and characteristics.
        - Display the shape of the dataset:
          ```python
          print(f"Banknote Shape: {banknote.shape}")
          ```
        - Show data types for each column:
          ```python
          print("Banknote Data Types:")
          print(banknote.dtypes)
          ```
        - View dataset information to check for missing values and column data types:
          ```python
          banknote.info()
          ```
          

2. **Data Preprocessing**
    - **Handling Missing Values**: Addressed missing values using the K-Nearest Neighbors (KNN) imputer to ensure complete data for model training.
      ```python
      imputer = KNNImputer()
      banknote_imputed = imputer.fit_transform(banknote)
      banknote = pd.DataFrame(banknote_imputed, columns=banknote.columns)
      ```
      This step fills in missing data to ensure complete inputs for model training.
      
    - **Data Splitting**: Divided the dataset into training and testing sets to facilitate model evaluation.
      ```python
      X = banknote.drop(columns='Class')
      y = banknote['Class']
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
      ```
      
    - **Data Standardization**: Scaled features using StandardScaler to improve model performance.
      ```python
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)
      ```

3. **Model Building**
    - **Logistic Regression Model**: Trained a logistic regression model on the standardized training set.
      ```python
      model = LogisticRegression()
      model.fit(X_train, y_train)
      ```
  
    - **Prediction**: Used the trained model to predict labels on the test set.
      ```python
      y_pred = model.predict(X_test)
      ```

4. **Evaluation Metrics**
     Evaluate model performance using metrics:
    - **Confusion Matrix**: Generated a confusion matrix to assess the model's performance in terms of true positives, false positives, true negatives, and false negatives.
      ```python
      print(confusion_matrix(y_test, y_pred))
      ```
      
    - **Accuracy Score**: Computed the accuracy score to provide a basic measure of the model's performance.
      ```python
      print(accuracy_score(y_test, y_pred))
      ```

5. **Model Interpretation and Analysis**
    - **Feature Correlation**: Display a heatmap for correlation between features:
      ```python
      sns.heatmap(banknote.corr(), annot=True, cmap="coolwarm")
      plt.title("Feature Correlation Matrix")
      plt.show()
      ```
      
    - **Performance Analysis**: Interpret the accuracy and confusion matrix to understand the model's performance and evaluate any areas for improvement or further analysis.
      ```python
      This methodology covers each primary step, from loading data to evaluating the model's performance. Adjust any steps based on specific requirements or updates in your code.
      ```

This structured methodology provides a comprehensive approach to developing, validating, and interpreting the Logistic Regression model on the Banknote Authentication dataset.


### Results Reflection

- **Linear Regression**
  - **R-squared**: 0.832 – Indicates that approximately 83.2% of the variance in the total count of bike rentals is explained by the model.
  - **Root Mean Square Error (RMSE)**: 953.95 – Average deviation of the model’s predictions from the actual values, emphasizing the model's precision in predicting total counts.
  - **Mean Absolute Error (MAE)**: 677.24 – Mean absolute deviation, highlighting how close predictions are to observed values on average.
  - **Prediction Example**: Predicted total count of bike rentals = 5904.62 vs. actual count = 6118, demonstrating a slight underprediction.
  - **Model Coefficients**: Reflects the contribution of each feature, with some features positively or negatively impacting predictions.
  
- **Logistic Regression**
  - **Accuracy**: 96.73% – High accuracy for classifying banknotes as forged or authentic, indicating strong model performance on the test set.
  - **Confusion Matrix**:
    - True Positives: 163 – Correctly identified forged banknotes.
    - True Negatives: 103 – Correctly identified authentic banknotes.
    - False Positives: 8 – Authentic banknotes misclassified as forged.
    - False Negatives: 1 – Forged banknotes misclassified as authentic.
  - **Model Coefficients**: Variance, skewness, and kurtosis exhibit strong negative influence, making forgery likely when increased; entropy, with a positive coefficient, increases likelihood of authenticity.

### Comparison of Regression Methods

- **Performance Metrics**:
  - Linear Regression is effective for predicting continuous values, providing insights into how feature variation affects rental counts. Logistic Regression, with high accuracy for categorical prediction, correctly classified most banknotes due to its effective separation of binary classes.
- **Interpretability**:
  - Linear Regression provides interpretable coefficients for continuous predictions, while Logistic Regression’s coefficients impact the probability of each class, allowing insight into each feature’s influence on classification.
- **Model Strengths and Weaknesses**:
  - Linear Regression effectively predicts continuous outcomes but struggles in binary classification. Logistic Regression’s design makes it ideal for binary classification but limited in scenarios needing continuous prediction.

### Limitations

- **Data Assumptions**:
  - Linear Regression assumes linearity and homoscedasticity, which may not fully capture real-world relationships in bike rentals. Logistic Regression assumes a binary target, making it unsuitable for non-binary outcomes.
- **Overfitting and Underfitting**:
  - Logistic Regression might slightly overfit if too finely tuned, while Linear Regression may underfit if relationships between features and target variable deviate from linearity.
- **Dataset Limitations**:
  - Limited feature sets in both datasets may omit influential factors (e.g., additional environmental conditions or unknown banknote characteristics), potentially impacting model performance and accuracy.

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





