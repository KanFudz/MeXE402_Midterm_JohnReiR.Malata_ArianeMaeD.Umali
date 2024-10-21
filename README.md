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
  - [VI. Logistic Regression Analysis](#vi-logistic-regression-analysis)
      - [Data Preprocessing](#data-preprocessing)
      - [Model Implementation](#model-implementation)
      - [Evaluation Metrics](#evaluation-metrics)
      - [Visualization](#visualization)
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
      <img src="https://github.com/user-attachments/assets/e6c3e5c0-a456-486d-92b8-8e894f92eab2" alt="Bike Sharing Dataset" style="width: 100%; height: 300px;">
    </td>
    <td width="50%">
      <div align="justify">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This Pair-Based Project aims to showcase our understanding of Linear and Logistic Regression through hands-on analysis of the Bike Sharing Dataset and Bank Note Authentication UCI data. The practical dataset exploration and selection of appropriate dependent variables are performed face-to-face. This repository documents the methodologies, performance of regression analyses, and interpretation of results.
      </div>
    </td>
    <td width="25%">
      <img src="https://github.com/user-attachments/assets/86b4a150-33f0-4fe8-b9b9-9c98f047efb7" alt="Investigator" style="width: 100%; height: 300px;">
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


## V. Linear Regression Analysis - Bike Rental Dataset

### Overview
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The project performs a Linear Regression analysis to predict bike rental counts based on environmental and calendar-based factors. The dataset includes features like temperature, humidity, weather conditions, and more. The analysis covers data exploration, preprocessing, model training, and evaluation.

### Exploratory Data Analysis (EDA)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The dataset consists of 731 observations and 16 features. Key variables include temperature, humidity, windspeed, and total bike rental counts. Columns are renamed for clarity, and the dataset is inspected for missing values and data types. Initial data exploration reveals correlations between weather conditions and bike rental counts.

### Data Preprocessing
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Missing values are handled using K-Nearest Neighbors (KNN) imputation, ensuring data consistency. Outliers and extreme values are checked to prevent skewing the model results. Continuous features are normalized to improve model performance. Features are split into training and test sets to evaluate the model's generalizability.

### Model Implementation
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A Linear Regression model is applied using the `Scikit-learn` library. The model uses features such as temperature, humidity, and windspeed to predict total bike rentals. The training process involves fitting the model to the dataset and evaluating its predictive power.

### Model Evaluation
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The model is evaluated using R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE). These metrics provide insights into how well the model explains the variability in bike rentals and the accuracy of its predictions. The coefficients of the model are interpreted to understand the impact of individual predictors.

#### Key Metrics:
- **R-squared:** Indicates the proportion of variance in bike rentals explained by the model.
- **MSE & MAE:** Measure the average errors in prediction, with lower values indicating better performance.

### Results
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The analysis shows a significant relationship between temperature and bike rentals, with a positive coefficient suggesting higher rentals during warmer conditions. Humidity and windspeed also contribute to predicting bike rentals, although to a lesser extent.

### Visuals:
- Distribution plots of key predictors and the target variable (`total_count`).
- Heatmaps to visualize correlations between variables.
- Residual plots to assess model assumptions like linearity and homoscedasticity.



## VI. Logistic Regression Analysis


<br>
<br>






## VII. Documentation
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [Text]
  </p>
<br>
<br>


## VIII. References
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [Text]
  </p>
<br>
<br>


## IX. Group Members

<div align="center">

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/2d9ebaa0-d550-4b60-856d-d2c98fb9f3d1" alt="Malata" style="height: 230px; float: left;"></td>
    <td><img src="https://github.com/yannaaa23/Testing/blob/49510d6b3798f3f40648deb0f4c8a903a48d1fc4/hello/IMG_20230605_215028_860.jpg" alt="Umali" style="height: 230px; float: left;"></td>
  </tr>
  <tr>
    <td align="center"><strong>Malata, John Rei R.</strong></td>
    <td align="center"><strong>Umali, Ariane Mae D.</strong></td>
  </tr>
</table>

</div>

<br>
<br>






# MIDTERMS Instruction link: https://github.com/MikkoDT/MeXE402_Midterm_4102
