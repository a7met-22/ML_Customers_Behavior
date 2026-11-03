# E-commerce Customer Churn: Exploratory Data Analysis

An **exploratory data analysis (EDA)** notebook that inspects an
e-commerce customer dataset in preparation for building a **churn
prediction model**. It checks data quality, measures skewness and
outliers, examines feature relationships and the target balance, and
investigates specific data-integrity and engagement questions, ending
each investigation with a written conclusion and action plan.

The notebook demonstrates a question-driven EDA workflow: every plot or
table is used to answer a specific question about the data, and the
findings are translated into concrete preprocessing decisions for the
modeling stage.

> **Project:** ML EDA: Customer Behavior (Churn)\
> **Notebook:** `ML-EDA-Customer_Behavior.ipynb`\
> **Domain:** Machine Learning · Exploratory Data Analysis · Data
> Quality · Customer Churn · Python · Pandas · Seaborn\
> **Environment:** Google Colab

------------------------------------------------------------------------

## 1. Project Overview

The goal is to understand a customer dataset well enough to prepare it
for predicting whether a customer **churns** (the target column is
`Churned`), and to document which data problems must be fixed first.

Instead of modeling immediately, the workflow works through a sequence
of checks:

1.  Load the dataset and inspect its structure, types, and first rows.
2.  Measure the percentage of missing values per column.
3.  Compute skewness and IQR-based outlier counts for every numeric
    feature.
4.  Visualize the most skewed features with boxplots and
    distributions.
5.  Inspect feature correlations and the balance of the target classes.
6.  Study the density of order value by age to detect impossible
    values.
7.  Investigate the suspicious records and decide whether to clip,
    transform, or drop them.
8.  Analyze how mobile app usage relates to engagement metrics.

The central engineering principle is:

> **Understand and clean the data before modeling it: every
> preprocessing decision should come from evidence found during
> exploration.**

------------------------------------------------------------------------

## 2. Key Features

### Data inspection

-   Loads the CSV file with Pandas.
-   Displays the first rows and the `info()` summary for types and
    non-null counts.

### Missing-value analysis

-   Computes counts and percentages of missing values per column.
-   Shows only columns that actually have missing values, sorted from
    most to least affected.

### Skewness and outlier analysis

-   Calculates skewness for all numeric columns.
-   Counts outliers per column using the **IQR rule** (1.5 × IQR beyond
    the quartiles).
-   Presents both in one sorted summary table.

### Visual diagnostics

-   Boxplot and distribution (histogram with KDE) for the four most
    skewed features, with the skewness value in each title.
-   Correlation heatmap across all numeric features.
-   Count plot and percentage breakdown of the target classes.
-   Hexbin density plot of average order value by age, used to avoid
    overplotting on a large dataset.

### Data-integrity investigation

-   Isolates records with an impossible age (above 100).
-   Displays their behavioral and spending metrics to decide whether
    they are real customers with a typo or junk records.

### Mobile engagement analysis

-   Compares mobile app usage with session duration, pages per
    session, and login frequency using regression plots.
-   Reports the correlation for each metric in the plot title.

### Documented conclusions

-   Markdown cells record the findings and a concrete action plan next
    to the code that produced them.

------------------------------------------------------------------------

## 3. Architecture

| Stage | Responsibility |
|-------|----------------|
| Data loading | Reads the churn CSV and shows structure, types, and sample rows. |
| Missing-value summary | Counts and ranks missing values by column. |
| Skewness and outlier report | Computes skewness and IQR outlier counts for numeric features. |
| Distribution plots | Boxplots and histograms for the most skewed features. |
| Correlation and target check | Heatmap of feature relationships and the churn class balance. |
| Density analysis | Hexbin plot of order value against age. |
| Integrity investigation | Examines records with impossible age values. |
| Engagement analysis | Regression plots of mobile app usage against engagement metrics. |
| Conclusions | Written findings and action plans in markdown. |

### Metrics used

| Metric | Purpose |
|--------|---------|
| Missing percentage | Decide how to impute or drop incomplete columns. |
| Skewness | Identify features that may need a transform. |
| IQR outlier count | Quantify extreme values per feature. |
| Pearson correlation | Detect linear relationships and redundancy. |
| Class balance | Decide whether resampling or class weights are needed. |

### High-level workflow

``` text
        Customer churn CSV
                |
                v
     Load + inspect structure
                |
                v
       Missing-value summary
                |
                v
   Skewness + IQR outlier report
                |
                v
     Distribution plots (top 4)
                |
                v
  Correlation heatmap + target balance
                |
                v
    Density analysis (order value vs age)
                |
                v
   Investigate impossible-age records
          /                 \
   Typo in a real       Junk / null
   customer record      record
        |                    |
        v                    v
   Clip / transform      Drop record
                |
                v
    Engagement analysis (mobile app)
                |
                v
   Conclusions + preprocessing plan
```

------------------------------------------------------------------------

## 4. Technical Highlights

### Question-driven exploration

Each step answers a specific question, such as whether high spenders
depend on age, whether impossible ages are data errors, or whether app
usage tracks engagement. The notebook then writes the answer and the
resulting action beside the code.

### IQR outlier counting

Outliers are counted per column with the interquartile range rule,
which does not assume a normal distribution and works well on skewed
data. Combining it with skewness shows which features are both extreme
and asymmetric.

### Density plot instead of a scatter plot

A hexbin plot with a logarithmic color scale shows where records are
concentrated even when thousands of points overlap, which makes sparse
extreme values stand out against the dense core.

### Findings documented in the notebook

The notebook's own written conclusions record these observations from
its analysis:

-   High order values appear across the full normal age range,
    suggesting that high spending is not driven by age.
-   A few records have ages around 150 and 200, which are impossible
    and are treated as data errors.
-   Age and average order value show near-zero linear correlation.
-   Mobile app usage correlates strongly and positively with session
    duration (0.76), pages per session (0.72), and login frequency
    (0.67).

### Planned preprocessing decisions

-   Filter or cap impossible ages (keep age at or below 100).
-   Apply a log transform to average order value to reduce the effect
    of very high spenders.
-   Decide between clipping and dropping the suspicious records based
    on whether their behavior looks normal or empty.

------------------------------------------------------------------------

## 5. Technology Stack

| Technology | Role |
|------------|------|
| Python | Core implementation language. |
| Google Colab | Notebook execution environment. |
| Pandas | Data loading, summaries, and filtering. |
| NumPy | Numeric selection and calculations. |
| SciPy | Statistics support. |
| Matplotlib | Plot layout, hexbin, and figures. |
| Seaborn | Boxplots, histograms, heatmap, count plot, regression plots. |

------------------------------------------------------------------------

## 6. Running the Notebook

### Requirements

The notebook is designed for Google Colab and expects the dataset file
`ecommerce_customer_churn_dataset.csv` to be available in the Colab
working directory.

Before running:

-   Open the notebook in Google Colab.
-   Upload the CSV to `/content/` or change `file_path` to its
    location.
-   Make sure the dataset contains the columns the notebook uses,
    including `Age`, `Average_Order_Value`, `Churned`,
    `Mobile_App_Usage`, `Session_Duration_Avg`, `Pages_Per_Session`,
    `Login_Frequency`, `Cart_Abandonment_Rate`, and
    `Customer_Service_Calls`.

### Execution order

The notebook cells should be run in this order:

1.  Load and inspect the data.
2.  Missing-value summary.
3.  Import plotting libraries (NumPy, Matplotlib, Seaborn).
4.  Skewness and outlier report, and its plots.
5.  Correlation heatmap and target balance.
6.  Hexbin density analysis and its conclusion.
7.  Investigation of impossible ages.
8.  Mobile engagement analysis and its conclusion.

Note that the skewness and plotting cells use NumPy, Matplotlib, and
Seaborn, which the notebook imports in a later cell; move the imports to
the top before running from top to bottom.

### Example output

The hexbin cell produces a density plot of order value against age,
where isolated points at impossible ages are visible away from the dense
cluster. The engagement cell produces three regression plots with the
correlation shown in each title.

------------------------------------------------------------------------

## 7. Configuration

Settings are defined inline so they can be adjusted without changing the
analysis structure.

Review these settings:

-   `file_path`: location of the dataset.
-   IQR multiplier (1.5) for outlier detection.
-   Number of features plotted (top 4 by skewness).
-   Age threshold for suspicious records (100).
-   Hexbin `gridsize` (30) and logarithmic color scale.
-   Metrics and behavioral columns listed in the analysis functions.

Do not commit private customer data to a public repository. If the
dataset contains real customer records, keep it out of version control.

------------------------------------------------------------------------

## 8. Limitations and Data Considerations

This notebook is an exploratory prototype. It documents findings and
plans but does not yet prepare data or train a model.

| Area | Current approach | Important limitation |
|------|------------------|----------------------|
| Saved results | Code and written conclusions are in the notebook. | The saved copy has no stored outputs (the first cell ended with a missing-file error), so the figures cited in the conclusions cannot be verified from the file alone. |
| Cell order | Imports appear where first needed. | NumPy, Matplotlib, and Seaborn are imported after the cells that use them, so a top-to-bottom run fails until the imports are moved up. |
| Unfinished work | Planned preprocessing is described in markdown. | The cleaning and transformation steps are not implemented, and several trailing cells are empty. |
| Age handling | Records above 100 are investigated. | The text proposes both filtering by age and applying a log transform to age, which conflicts with the earlier plan to log-transform order value instead. |
| Variable type | Mobile app usage is described as binary. | The plot labels it as a frequency, and a linear regression line is drawn against it; its type should be clarified before interpreting the correlations. |
| Correlation | Pearson correlation and regression plots. | Correlation is linear and does not show causation or non-linear relationships. |
| Target relationship | Class balance is plotted. | Relationships between individual features and churn are not yet examined. |
| Missing values | Reported per column. | No handling strategy has been chosen or tested. |
| Outliers | IQR counts and visual checks. | Flagged values are not removed, capped, or tested for their effect on a model. |
| Reproducibility | Single notebook with inline settings. | No fixed environment file or documented dataset source. |

### Recommendations before modeling

-   Move all imports to the first cell and re-run the notebook with the
    dataset to save its outputs.
-   Resolve the age-handling plan and implement it in code.
-   Confirm the type and meaning of mobile app usage.
-   Analyze each feature against the churn target, not only against
    other features.
-   Choose and document a strategy for missing values and outliers.
-   Split the data before fitting any transform to avoid leakage.
-   Decide how to handle class imbalance after checking the target
    distribution.
-   Document the dataset source and column definitions.

------------------------------------------------------------------------

## 9. Repository Structure

``` text
.
├── ML-EDA-Customer_Behavior.ipynb
└── README.md
```

The dataset file is not included and should be added separately, only if
it contains no private customer data.

------------------------------------------------------------------------

## 10. Possible Future Improvements

-   Implement the planned cleaning steps (age filtering and log
    transform) in a reusable preprocessing function.
-   Analyze every feature against `Churned` with appropriate tests and
    plots.
-   Add categorical feature analysis and encoding decisions.
-   Add missing-value imputation strategies and compare their effect.
-   Measure how outlier treatment changes model performance.
-   Build baseline models such as logistic regression and tree-based
    classifiers.
-   Evaluate with metrics suited to churn, such as recall, precision,
    and ROC-AUC, with cross-validation.
-   Add feature importance and explainability analysis.
-   Package the pipeline so preprocessing and modeling are
    reproducible.

------------------------------------------------------------------------

## 11. Key Takeaway

This project demonstrates a disciplined first stage of a machine
learning project: inspecting data quality, quantifying skewness and
outliers, discovering impossible values, and recording the preprocessing
decisions that follow.

The most important design lesson is that **a good model starts with a
well-understood dataset: exploration should end with documented,
evidence-based decisions about what to clean, transform, and test
before any model is trained.**
