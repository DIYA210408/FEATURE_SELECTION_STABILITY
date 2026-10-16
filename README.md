# Feature Selection Stability Under Missing Data

## 1. Project Overview

This project studies how missing data affects **feature-selection stability** and **classification performance**. It investigates whether different imputation and feature-selection methods continue to select similar features as the amount and mechanism of missingness change.

The experiments use the **Breast Cancer Wisconsin (Diagnostic)** dataset and compare results under two missing-data mechanisms: **Missing Completely at Random (MCAR)** and **Missing at Random (MAR)**.

## 2. Objectives

- Introduce controlled levels of missing data into the dataset.
- Compare Mean, Median, and K-Nearest Neighbors (KNN) imputation.
- Compare six feature-selection methods using a fixed top-10 feature set.
- Measure whether selected features remain consistent across repeated runs.
- Evaluate the predictive performance of a classifier trained using the selected features.
- Compare feature-selection stability and predictive performance under MCAR and MAR missingness.
- Save experiment results, summary tables, and plots for analysis.

## 3. Dataset

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), loaded through `sklearn.datasets.load_breast_cancer`.
- **Samples:** 569
- **Input features:** 30 numerical features
- **Target:** Binary classification (malignant or benign).
- **Train/test split:** 80% training and 20% testing, using stratification and `random_state=42`.

The split is created before the missingness experiments. Imputation is fitted on the training data and then applied to the test data.

## 4. Experimental Setup

### Missingness levels

The experiments consider the following proportions of missing feature values:

`0%, 5%, 10%, 20%, 30%, 40%`

### Missingness mechanisms

- **MCAR — Missing Completely at Random:** Missing values are introduced without using the observed feature values to determine which values are missing.
- **MAR — Missing at Random:** The implementation makes the probability of a value being missing depend on the observed `mean radius` feature. The `mean radius` driver feature itself is kept observed.

The MAR experiment is an extension of the MCAR experiment. Results should be compared using the saved output tables and plots rather than assuming one mechanism is always worse.

### Imputation methods

1. **Mean imputation:** Replaces missing values with the training feature mean.
2. **Median imputation:** Replaces missing values with the training feature median.
3. **KNN imputation:** Estimates missing values from similar observations.

### Feature-selection methods

The experiment selects **10 features** from the 30 available features using each of these methods:

1. **Pearson correlation:** Ranks features by the absolute correlation with the target.
2. **ANOVA F-test:** Uses `f_classif` to score feature–target relationships.
3. **Mutual Information:** Estimates the dependency between each feature and the target.
4. **Random Forest importance:** Ranks features using a Random Forest classifier's feature-importance scores.
5. **LASSO:** Uses regularized logistic-model coefficients to rank features.
6. **Recursive Feature Elimination (RFE):** Recursively removes less useful features to retain the requested number.

Each combination of missingness level, imputation method, and feature-selection method is repeated **30 times** with different random seeds.

For the MCAR experiment, this gives 6 missingness levels × 3 imputation methods × 6 feature-selection methods × 30 repetitions = **3,240 experimental runs**. The same design is used for the MAR extension.

## 5. Stability Evaluation

Stability measures how consistently a feature-selection method chooses the same features across repeated runs.

- **Jaccard index:** Measures the overlap between two selected feature sets. A value closer to 1 indicates greater overlap.
- **Kuncheva index:** Measures overlap while accounting for the number of available features and the selected subset size. Higher values indicate greater stability; its interpretation depends on the index's defined range and setup.

The notebook aggregates these scores across repeated runs for each experimental condition and records average and standard-deviation values where available.

## 6. Predictive Performance Evaluation

After feature selection, a **StandardScaler + Logistic Regression** pipeline is trained using the selected training features and evaluated on the held-out test set.

The recorded metrics are:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**

The notebook also combines stability and performance summaries to examine whether more stable feature selection is associated with better predictive accuracy. Correlation analysis and scatter plots are used for this comparison.

## 7. Analysis and Visualizations

The notebook creates summary tables and plots to examine:

- Jaccard stability across missingness levels.
- Kuncheva stability across missingness levels.
- Accuracy under different imputation and feature-selection methods.
- The relationship between stability and predictive accuracy.
- The effect of imputation method on stability.
- Differences between feature-selection methods.
- The most frequently selected features and their recurrence as missingness increases.
- A heatmap comparing imputation/feature-selection combinations.
- MCAR-versus-MAR comparisons for stability and predictive performance.

Interpret the findings from the generated CSV files and figures. No single method should be described as the best unless the saved results support that conclusion.

## 8. Main Tools and Libraries

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- Google Colab / Jupyter Notebook

## 9. Results and Output Files

The notebook writes experiment outputs into folders such as `results/`, `results_mar/`, `results_comparison/`, `plots/`, and `plots_comparison/`, depending on which cells have been executed.

Expected output categories include:

| Output | Purpose |
|---|---|
| `selected_features.csv` | Selected feature names for each experimental condition and repetition |
| `stability_results.csv` | Jaccard and Kuncheva stability summaries for MCAR |
| `performance_results.csv` | Predictive metrics recorded across repetitions |
| `performance_summary.csv` | Mean and standard deviation of predictive metrics |
| `master_stability_results.csv` | Consolidated stability table |
| `master_performance_results.csv` | Consolidated performance table |
| `master_results.csv` | Combined stability and performance results |
| `mar_stability_results.csv` | Stability results for the MAR experiment |
| `mar_performance_results.csv` | Predictive performance results for the MAR experiment |
| `mcar_mar_stability.csv` | Combined stability data for MCAR-versus-MAR analysis |
| `mcar_mar_performance.csv` | Combined predictive performance data for MCAR-versus-MAR analysis |
| `*.png` figures | Saved plots used to compare methods and summarize results |

The exact files available depend on which notebook cells ran successfully. Check the output folders before submission and include the generated CSVs and figures, not just the notebook.

## 10. How to Run

1. Open the notebook in Google Colab or a Jupyter environment.
2. Install any missing dependencies if prompted.
3. Run the notebook cells in order, beginning with imports and dataset preparation.
4. Allow the MCAR and MAR experiments to complete; repeated experiments may take time.
5. Check that the output CSVs and plots were created in their respective folders.
6. Review the summary tables and figures before writing conclusions.
7. Keep the notebook, result CSVs, figures, and this README together when packaging the project.

## 11. Limitations and Considerations

- Results are specific to this dataset, split, random seeds, missingness designs, and model choices.
- MAR is simulated using `mean radius` as an observed driver; it represents one MAR design, not every possible MAR situation.
- The selected top-10 features and the resulting metrics can change with different datasets, parameters, or random seeds.
- Stability and predictive performance are different properties: consistent feature selection does not automatically guarantee the highest classification score.
- Verify that all output files were generated successfully and that the reported tables correspond to the final executed notebook before submitting or publishing results.

## 12. Project Summary

The project builds a repeated experimental pipeline to compare **three imputation methods** and **six feature-selection methods** under increasing missingness. It evaluates selected-feature consistency using Jaccard and Kuncheva indices, measures classification performance using Logistic Regression, and extends the analysis from MCAR to a simulated MAR mechanism. The resulting tables and visualizations support a systematic comparison of robustness under missing data.
