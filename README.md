# Diabetes Dataset Analysis and Classification

This project performs exploratory data analysis (EDA) and classification modeling on the **PIMA Indian Diabetes Dataset**. It visualizes the data, detects and handles outliers, and uses various classification algorithms like Logistic Regression, Support Vector Machine, K-Nearest Neighbors, Random Forest, Naive Bayes, and Gradient Boosting to predict diabetes occurrence.

## Table of Contents
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Outlier Removal](#outlier-removal)
- [Model Building](#model-building)
- [Results](#results)
- [Usage](#usage)

## Dataset
The dataset used is the **PIMA Indian Diabetes Dataset**. It includes 768 instances of medical records with the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (0 = no diabetes, 1 = diabetes)

## Dependencies

To run this project, you need to install the following Python libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```
## Project Structure 
The project is structured as follows:

```bash
├── diabetes_analysis.ipynb   # Jupyter notebook for EDA and model building
├── diabetes.csv              # Dataset used in the analysis
└── README.md                 # Project documentation
```

## Exploratory Data Analysis (EDA)

1. **Dataset Overview:** The dataset is loaded, and basic statistics such as mean, median, and distribution of each variable are displayed.
2. **Missing Values:** The dataset is checked for missing values, and a **histogram** is plotted for all features to visualize the data distribution.
3. **Correlation Matrix:** A heatmap is generated to show the correlation between features.
4. **Outcome Distribution:** A count plot is displayed to understand the distribution of diabetic vs. non-diabetic patients.


## Outlier Removal
- **Boxplot Analysis:** Box plots are used to detect outliers for features such as insulin, BMI, blood pressure, and pedigree function.
- **Interquartile Range (IQR)** is used to remove outliers from the dataset.

## Model Building 
After performing EDA and handling outliers, the cleaned data is used for classification modeling. The following machine learning algorithms are applied:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **K-Nearest Neighbors (KNN)**
4. **Random Forest Classifier**
5. **Naive Bayes Classifier**
6. **Gradient Boosting Classifier**


Each model is evaluated based on:

- **Accuracy:** Percentage of correctly classified instances.
- **ROC AUC Score:** The area under the ROC curve to measure the model’s ability to distinguish between positive and negative classes.
## Cross Validation
- **Cross Validation** is used for each model to calculate metrics such as True Positive, True Negative, False Positive, and False Negative.
## Results
After applying all models, the following results are observed:

- **Random Forest Classifier** gives the highest accuracy (98%) and ROC AUC score (97%).
- A bar chart is plotted comparing the performance of all algorithms in terms of **accuracy** and **ROC AUC** score.

## Usage
1- **Clone the repository:**
```
git clone https://github.com/your-username/diabetes-classification.git
cd diabetes-classification
```

2- **Install dependencies:**
``
pip install pandas numpy seaborn matplotlib scikit-learn
```
3- **Run the Jupyter Notebook:**
```
jupyter notebook diabetes_analysis.ipynb
```
4- **Load the dataset:***
 Make sure the dataset **diabetes.csv** is in the same directory as the notebook or update the file path accordingly in the code.









