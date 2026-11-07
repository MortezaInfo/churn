# Telecom Customer Churn Prediction

A machine learning project that explores how customer account and service data can be used to identify telecom customers at risk of churn. The project focuses on the practical challenges of a binary classification problem, especially the effect of class imbalance on churn detection.

## Project Highlights

- Explores a telecom customer dataset containing **7,043 records** and **19 customer features**.
- Cleans and prepares customer data, including handling invalid `TotalCharges` values, encoding categorical features, and scaling numeric features.
- Trains a **Logistic Regression** classifier and evaluates it with precision, recall, F1-score, and accuracy.
- Compares class weighting, random under-sampling, random over-sampling, and SMOTE to study their impact on detecting churn.
- Uses visual exploration and classification reports to make model trade-offs easier to interpret.

## Results

The notebook reports the following F1-scores for the churn class (class `1`):

| Approach | Churn F1-score |
| --- | ---: |
| Logistic Regression, no balancing | 0.59 |
| Class weighting | 0.64 |
| Random under-sampling | 0.75 |
| Random over-sampling | 0.78 |
| SMOTE | 0.80 |

These are exploratory results recorded in the notebook, not production performance estimates. In the current resampling experiments, balancing is performed **before** the train/test split, which can introduce data leakage and make the reported scores optimistic. For a reliable comparison, split the original data first, apply resampling to the training set only, and evaluate on an untouched test set with the original class distribution.

## Dataset

The included file, [`IT_customer_churn.csv`](./IT_customer_churn.csv), has 7,043 customer records. It contains customer demographics, account tenure, subscribed services, contract and payment details, monthly and total charges, and the `Churn` target.

## Getting Started

### Requirements

- Python 3
- Jupyter Notebook
- NumPy
- pandas
- Matplotlib
- Plotly
- scikit-learn
- imbalanced-learn

Install the packages with:

```bash
python -m pip install numpy pandas matplotlib plotly scikit-learn imbalanced-learn jupyter
```

### Run the notebook

From the project directory, start the notebook:

```bash
jupyter notebook "Classification - Churn Modeling ( Telecom ).ipynb"
```

Run the cells from top to bottom. The notebook reads the included `IT_customer_churn.csv` file.

## Workflow

1. Load and inspect the customer data.
2. Explore churn distribution, customer characteristics, and service patterns.
3. Clean data and prepare categorical and numeric features.
4. Train and evaluate Logistic Regression models.
5. Compare imbalance-handling strategies using class-level metrics.

## Repository Contents

```text
.
├── Classification - Churn Modeling ( Telecom ).ipynb
├── IT_customer_churn.csv
└── README.md
```

## Skills Demonstrated

Python, pandas, NumPy, data cleaning, exploratory data analysis, data visualization, feature preprocessing, scikit-learn, imbalanced classification, model evaluation, and Jupyter Notebook.
