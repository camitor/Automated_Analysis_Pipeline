# Automated Analysis Pipeline Exploration

## Overview

This repository contains my Week 3 **Automated Analysis Pipeline Exploration** for the Health Data Science program at Dartmouth College.

The purpose of this project is to examine, execute, and extend a structured analytical pipeline for a binary health outcome. The notebook demonstrates how reproducible programming practices, conditional logic, exploratory data analysis, inferential statistics, and supervised machine learning can be integrated into a single analytical workflow.

The original notebook provided a structured pipeline that adapts portions of the analysis based on dataset characteristics and user-defined analytical choices. I extended the workflow with additional automated analyses focused on class balance, standardized effect size, logistic regression interpretation, and model generalization.

---

### Files

- **`Automated_Analysis_Pipeline_Notebook.ipynb`** — Completed Google Colab notebook containing the full analytical workflow and modifications.
- **`diabetes.csv`** — Dataset used to execute the analysis.
- **`README.md`** — Project documentation, execution instructions, assumptions, and limitations.

---

## Dataset

The notebook analyzes the **Pima Indians Diabetes Database**, a health dataset containing diagnostic measurements used to examine a binary diabetes outcome.

The dataset contains **768 observations** and the following variables:

| Variable | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Glucose measurement |
| `BloodPressure` | Blood pressure measurement |
| `SkinThickness` | Skin thickness measurement |
| `Insulin` | Insulin measurement |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age |
| `Outcome` | Binary diabetes outcome (`0` or `1`) |

`Outcome` is used as the target variable for the supervised machine learning analyses.

The outcome distribution contains:

- **500 observations (65.1%)** with `Outcome = 0`
- **268 observations (34.9%)** with `Outcome = 1`

The automated class-balance assessment therefore identifies the positive class as the minority class and describes the dataset as having moderate class imbalance.

---

## Analytical Workflow

The notebook follows a structured health data science workflow consisting of:

1. **Data ingestion and validation**
2. **Data quality assessment**
3. **Exploratory data analysis**
4. **Inferential statistical analysis**
5. **Supervised machine learning**
6. **Model interpretation and evaluation**

Conditional logic and reusable functions are used throughout the notebook so that portions of the workflow can respond to dataset characteristics or user-defined analytical choices.

For example, the workflow:

- selects an appropriate data-ingestion method based on file type,
- identifies numeric predictors,
- evaluates missing values,
- validates requirements for statistical and machine learning analyses,
- responds to outcome prevalence,
- evaluates the magnitude and direction of standardized group differences,
- applies model-specific preprocessing,
- selects a logistic regression classification threshold using training-set cross-validation, and
- evaluates differences between training and test performance.

User-defined parameters such as the target variable, variables selected for statistical testing, significance level, classification threshold, and decision-tree scoring metric also influence the analytical workflow.

---

## Exploratory Data Analysis

The exploratory analysis examines the distributions and relationships among the clinical predictors and diabetes outcome.

The notebook includes:

- Pearson correlation
- Spearman correlation
- Phi-K correlation
- Histograms
- Distribution plots
- Boxplots
- Violin plots
- Scatterplots
- Joint plots
- Strip plots
- Swarm plots
- Outcome-specific comparisons

The exploratory analysis shows that relationships vary depending on the measure of association used and that several predictors display skewness, potential outliers, or differences across outcome groups.

The outcome-balance assessment also identified moderate class imbalance. This informed the later decision to evaluate machine learning performance using **recall, precision, F1 score, and ROC AUC in addition to accuracy**, rather than relying on accuracy alone.

---

## Inferential Statistics

The inferential component of the notebook demonstrates several approaches for evaluating hypotheses involving continuous and categorical variables.

Methods include:

- **One-sample t-test**
- **Welch's independent-samples t-test**
- **One-way ANOVA**
- **Chi-square test of independence**

The analytical method depends on the research question and structure of the variables being evaluated. User-defined variables, thresholds, and significance levels allow these analyses to be adapted without rewriting the complete testing workflow.

An additional automated standardized effect-size analysis was added to complement significance testing and provide information about the magnitude of observed group differences.

---

## Supervised Machine Learning

The notebook evaluates two supervised binary classification models:

### Logistic Regression

The logistic regression workflow uses a scikit-learn `Pipeline` to:

1. impute missing predictor values using training-set medians,
2. standardize predictors,
3. fit the logistic regression classifier, and
4. apply the fitted preprocessing and model to test data.

The notebook evaluates logistic regression using multiple classification metrics and ROC AUC.

A classification cutoff is also selected using **5-fold stratified cross-validation on the training data** using Youden's J statistic. The selected threshold is subsequently applied to test-set probabilities.

### Decision Tree

The decision tree workflow uses:

- median imputation,
- `DecisionTreeClassifier`,
- `GridSearchCV`,
- 5-fold stratified cross-validation, and
- F1 score as the model-selection criterion.

The tuning procedure evaluates combinations of tree depth, minimum leaf size, maximum leaf nodes, and class weighting before refitting the selected model on the complete training set.

Feature importance and a visualization of the tuned decision tree are also included.

---

## Model Evaluation

Model performance is evaluated using:

- Accuracy
- Recall
- Precision
- F1 score
- ROC AUC
- Confusion matrices
- Training-versus-test performance

On the test set, the automated model comparison produced:

| Model | Accuracy | Recall | Precision | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.749 | 0.580 | 0.662 | 0.618 | **0.855** |
| Decision Tree | **0.835** | **0.728** | **0.787** | **0.756** | 0.811 |

At the default classification threshold, the decision tree achieved higher accuracy, recall, precision, and F1 score. Logistic regression, however, achieved the higher ROC AUC, indicating better overall discrimination across possible classification thresholds.

These results illustrate why classification models should be evaluated using multiple performance measures rather than a single metric.

---

# Pipeline Modifications

The assignment required modification of at least three components of the provided notebook. The completed workflow includes several modifications and four additional automated analytical extensions.

## Reproducible Random Sampling

The random record-inspection step was modified to use:

```python
random_state=101
```

Using a fixed random state ensures that the same observations are selected after restarting and rerunning the notebook. This improves reproducibility while retaining the benefit of inspecting observations from different locations in the dataset.

---

## Automated Analysis 1: Outcome-Balance Assessment

An automated class-balance assessment was added to evaluate the distribution of the binary outcome.

The analysis automatically:

- counts observations in each outcome class,
- calculates class prevalence,
- identifies the minority class, and
- provides a descriptive assessment of class imbalance.

For this dataset, `Outcome = 1` represents **34.9%** of observations and is identified as the minority class.

This analysis provides context for subsequent machine learning evaluation. Because overall accuracy can obscure performance on a minority class, recall, precision, and F1 score are considered alongside accuracy.

---

## Automated Analysis 2: Standardized Effect-Size Assessment

An automated Cohen's d assessment was added to complement the independent-samples statistical comparison.

The analysis:

- validates the selected variable and outcome groups,
- calculates Cohen's d,
- reports its absolute magnitude,
- provides a descriptive magnitude classification, and
- identifies the direction of the group difference.

For BMI, the analysis produced:

```text
Cohen's d = -0.697
Absolute effect size = 0.697
Descriptive magnitude = Moderate
```

The negative sign reflects the direction of the group comparison, with `Outcome = 1` having the higher mean BMI.

This complements the statistically significant Welch's t-test by demonstrating that the difference is also moderate in standardized magnitude according to the descriptive guidelines used in the automated assessment.

Because the data are observational, this result represents an **association rather than evidence of a causal effect**.

---

## Automated Analysis 3: Logistic Regression Interpretation

The logistic regression workflow was extended to automatically extract the fitted model coefficients and convert them to odds ratios.

The analysis:

- extracts coefficients from the fitted pipeline,
- matches coefficients with predictor names,
- calculates odds ratios,
- identifies the direction of each association, and
- ranks predictors by the magnitude of their standardized coefficients.

Because predictors are standardized before model fitting, each odds ratio corresponds to the estimated change in the odds of the positive outcome associated with a **one-standard-deviation increase** in the predictor, conditional on the other predictors in the model.

For example, Glucose had the largest standardized logistic regression coefficient in the fitted model, with an estimated odds ratio of approximately **2.396**.

These values describe model-estimated associations and should not be interpreted as causal effects.

---

## Automated Analysis 4: Decision Tree Generalization Assessment

The tuned decision tree evaluation was extended with an automated comparison of training and test performance.

The analysis calculates the training-to-test difference for:

- Accuracy
- Recall
- Precision
- F1 score

It then provides a descriptive assessment of the size of each generalization gap and automatically identifies the largest difference.

The largest observed generalization gap occurred for **recall**, with:

```text
Training recall = 0.877
Test recall     = 0.728
Gap             = 0.149
```

The analysis therefore identified a relatively large decline in recall from the training data to unseen test data, suggesting possible overfitting for at least some aspects of model performance.

The gap thresholds used in this assessment are descriptive diagnostics rather than universal statistical criteria for establishing overfitting.

---

## Conditional Logic and Automation

Conditional logic is an important component of the notebook.

Examples include:

- selecting a file-reading method based on the uploaded file format,
- checking whether required variables exist,
- identifying numeric predictors,
- detecting missing values,
- validating binary outcome coding,
- validating statistical-test requirements,
- applying scaling only when appropriate for logistic regression,
- identifying the minority outcome class,
- categorizing effect-size magnitude,
- identifying the direction of logistic regression coefficients,
- evaluating training-to-test generalization gaps, and
- selecting responses based on observed analytical results.

These components allow the notebook to adapt to characteristics of the data and user-defined analytical parameters rather than functioning only as a fixed sequence of commands.

---

# Reproducibility

The completed notebook was tested by restarting the Google Colab session and executing the workflow from beginning to end.

To improve reproducibility:

- a fixed random state is used where appropriate,
- the train/test split uses a fixed random seed,
- class proportions are preserved using stratified sampling,
- preprocessing is fitted using training data rather than the full dataset,
- cross-validation is performed using reproducible stratified folds,
- machine learning preprocessing and models are combined using scikit-learn pipelines, and
- reusable functions are used for repeated analytical tasks.

These practices reduce unintended variation between executions and help limit information leakage between training and test data.

---

# Running the Analysis

## Option 1: Google Colab

1. Open the notebook in Google Colab using the link or badge below.
2. Upload the dataset when prompted by the notebook.
3. Confirm that the expected dataset is loaded.
4. Select **Runtime → Restart session and run all**.
5. Allow all notebook cells to execute sequentially.
6. Review the generated statistical, graphical, and machine learning outputs.

### Open in Google Colab

Add the repository-specific Colab URL here after the notebook is uploaded to GitHub:

```markdown
[![Open In Colab](https://colab.research.google.com/drive/11IhrVcgXEBxtcnj4Fs22P-vczJXqGouR?usp=sharing))
```

---

## Option 2: Local Jupyter Environment

Clone the repository:

```bash
git clone YOUR-REPOSITORY-URL
cd YOUR-REPOSITORY-NAME
```

Launch Jupyter Notebook or JupyterLab and open:

```text
Automated_Analysis_Pipeline_Notebook.ipynb
```

The notebook was developed for Google Colab, so minor changes to the file-upload step may be required when running it in a local Jupyter environment.

---

# Required Software and Libraries

The analysis uses Python and the following major libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `statsmodels`
- `scikit-learn`
- `phik`

A Google Colab environment is recommended for reproducing the submitted workflow.

If running locally, required packages can be installed using:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn phik
```

Exact package versions may differ between environments. The submitted notebook was validated using the Google Colab environment in which the final analysis was executed.

---

# Assumptions and Limitations

Several assumptions and limitations should be considered when interpreting the analysis.

### Observational data

The dataset is observational. Statistical associations, regression coefficients, odds ratios, feature importance scores, and other model outputs should **not be interpreted as evidence of causal relationships**.

### Generalizability

The dataset represents a specific study population and may not generalize to other demographic or clinical populations. Model performance in this notebook therefore should not be interpreted as evidence of clinical performance in a broader patient population.

### Class imbalance

The positive outcome represents 34.9% of observations. Although the imbalance is not extreme, accuracy alone may provide an incomplete description of predictive performance. Recall, precision, F1 score, ROC AUC, and confusion matrices are therefore considered alongside accuracy.

### Missing-data workflow

The provided pipeline contains logic for identifying clinically impossible zero values and treating them as missing before machine learning imputation. In the dataset version used for the completed analysis, the evaluated clinical measurements did not contain zero-coded missing values requiring replacement during the final execution.

The machine learning pipelines nevertheless retain median-imputation steps so that preprocessing remains part of the fitted workflow and can accommodate missing predictor values if present.

### Imputation

Median imputation is a practical preprocessing strategy but does not account for uncertainty in missing values and may not be appropriate under all missing-data mechanisms.

### Statistical assumptions

Inferential procedures rely on assumptions associated with their respective statistical tests. Statistical significance alone does not indicate practical or clinical importance, which motivated the addition of standardized effect-size assessment.

### Descriptive thresholds

Thresholds used to categorize class imbalance, effect-size magnitude, and model generalization gaps are intended as descriptive analytical aids. They should not be interpreted as universal clinical or statistical standards.

### Model evaluation

Performance estimates are based on the selected train/test split and cross-validation procedures. Performance on new populations or external datasets may differ.

### Decision tree feature importance

Decision tree feature importance represents a feature's contribution to impurity reduction within the fitted tree. It does not measure causal influence or direct clinical effect on diabetes risk.

---

# Key Takeaways

This project demonstrates how an analytical notebook can be structured as a reproducible and partially adaptive health data science workflow rather than a collection of isolated analyses.

The completed workflow integrates:

- automated data ingestion and validation,
- exploratory data analysis,
- conditional logic,
- inferential statistics,
- standardized effect-size assessment,
- reproducible preprocessing,
- supervised machine learning,
- cross-validation,
- threshold selection,
- model interpretation, and
- generalization assessment.

The additional automated analyses extend the provided pipeline beyond statistical significance and overall predictive accuracy by considering **class balance, effect magnitude, model interpretability, and performance on unseen data**.

Together, these components demonstrate how analytical decisions can respond to both dataset characteristics and user-defined goals while maintaining a reproducible workflow.
