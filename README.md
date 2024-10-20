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


## II. Introduction

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This project is part of the **MeXE 402 - Mechatronics Engineering Elective 2: Data Science and Machine Learning** course midterm exam. The goal of this project is to apply **Linear Regression** and **Logistic Regression** techniques to analyze real-world datasets, including the **Bike Sharing Dataset** and the **Bank Note Authentication UCI data**. These two models are essential tools in data science for predicting and classifying outcomes based on input variables.

## Overview of Linear Regression

**Linear Regression** is a statistical method used to model the relationship between a dependent variable and one or more independent variables. The goal is to fit a straight line that best represents the relationship between the variables by minimizing the sum of squared differences between the observed values and the predicted values. It's commonly used for prediction and forecasting tasks, such as:

- Predicting sales based on advertising expenditure.
- Estimating house prices from features like area, number of rooms, etc.

### Linear Regression Equation

The general equation for a linear regression model can be represented as:

**$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon
$$**

Where:
- <b>y</b> is the dependent variable (e.g., number of bike rentals).
- <b>X<sub>1</sub>, X<sub>2</sub>, ..., X<sub>n</sub></b> are the independent variables.
- <b>&beta;<sub>0</sub></b> is the intercept.
- <b>&beta;<sub>1</sub>, &beta;<sub>2</sub>, ..., &beta;<sub>n</sub></b> are the coefficients.
- <b>&epsilon;</b> is the error term.

In this project, we will apply Linear Regression to the **Bike Sharing Dataset** to predict the number of bike rentals based on weather conditions, season, and other factors.

## Overview of Logistic Regression

**Logistic Regression** is used when the dependent variable is binary (e.g., yes/no, true/false, or 0/1). It estimates the probability that a given input belongs to a specific category. The model outputs probabilities, which are then thresholded to classify the input into one of two classes.

### Logistic Regression Equation

For logistic regression, the model predicts the probability of a binary outcome, expressed as:

**$$
P(Y=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n)}}
$$**

Where:
- <b>P(Y=1 | X)</b> is the probability that the dependent variable <b>Y</b> equals 1 given the independent variables <b>X</b>.
- <b>e</b> is the base of the natural logarithm.
- <b>&beta;<sub>0</sub>, &beta;<sub>1</sub>, &beta;<sub>2</sub>, ..., &beta;<sub>n</sub></b> are the model coefficients.

In this project, we will apply Logistic Regression to the **Bank Note Authentication UCI Dataset** to classify banknotes as either authentic or counterfeit based on their features.


## III. Dataset Description
<p align="justify"> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Details about the datasets used.
  </p>
<br>
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
