<h1 align="center">Midterm: Linear Regression and Logistic Regression</h1>
<h2 align="center">MExEE 402 - MExE Elective 2</h2>
<br>

## Table of Contents
  - [I. Abstract](#i-abstract)
  - [II. Introduction](#ii-introduction)
  - [III. Dataset Description](#iii-dataset-description)
  - [IV. Project Objectives](#iv-project-objectives)
  - [V. Linear Regression Analysis](#v-linear-regression-analysis)
      - [Data Preprocessing](#data-preprocessing)
      - [Model Implementation](#model-implementation)
      - [Evaluation Metrics](#evaluation-metrics)
      - [Interpretation](#interpretation)
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

<img align="left" src="https://th.bing.com/th/id/OIP.WAr6BRiHVI6Zo5bbzDjisgAAAA?rs=1&pid=ImgDetMain" style="height: 480px;"></p>

- **Infromation:**
    <p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Bike sharing systems are modern versions of traditional rentals, automating membership, rental, and returns. Users can rent and return bikes at different locations, with over 500 programs and 500,000 bicycles worldwide. These systems are gaining attention for their role in addressing traffic, environmental, and health concerns. Unlike other transport services, bike sharing records precise data on travel duration, and departure/arrival points, effectively turning the system into a virtual sensor network that can help monitor city mobility and detect important events.
    </p>
  <br>

- **Description:**
  <p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This dataset includes data on bike rentals across multiple seasons, with features capturing various environmental and temporal conditions. The dataset comprises several variables such as temperature, humidity, wind speed, season, holiday, working day, and weather situations. Each row represents daily rentals, which are influenced by these factors. The data provides an excellent opportunity to predict the number of rentals based on environmental factors. The main target variable is the number of rentals (count), which we aim to predict using <b>Linear Regression</b>.
    </p>
  <br>
  

<img align="right" src="https://thumbs.dreamstime.com/b/citibike-bicycle-share-new-york-april-bicycles-citibank-s-program-await-riders-new-york-city-photo-taken-april-40609911.jpg" style="height: 400px;"></p>

- **Attributes:** •	


  <br>


## IV. Project Objectives
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; What you aim to achieve with your analyses.
  </p>
<br>
<br>


## V. Linear Regression Analysis
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Perform Linear Regression using the selected continuous dependent variable
  </p>
<br>
<br>

### Data Preprocessing
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Handle missing values, outliers, and normalize data if necessary.
  </p>
<br>
<br>

### Model Implementation
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Use appropriate libraries (e.g., Scikit-learn in Python).
  </p>
<br>
<br>

### Evaluation Metrics
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Calculate R-squared, Mean Squared Error, etc.
  </p>
<br>
<br>


### Interpretation
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Explain the significance of coefficients and the model's predictive power.
  </p>
<br>
<br>



## VI. Logistic Regression Analysis
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [Text]
  </p>
<br>
<br>

### Data Preprocessing
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Encode categorical variables, balance classes if needed.
  </p>
<br>
<br>

### Model Implementation
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Use appropriate libraries.
  </p>
<br>
<br>

### Evaluation Metrics
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Calculate Accuracy
  </p>
<br>
<br>

### Visualization
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Plot confusion matrices
  </p>
<br>
<br>

### Interpretation
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Discuss the model's ability to classify and the importance of features.
  </p>
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
