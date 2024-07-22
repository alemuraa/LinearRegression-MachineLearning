# Linear Regression 

This assignment consists of three main tasks:

1. **Task 1: Get Data**
2. **Task 2: Fit a Linear Regression Model**
3. **Task 3: Test the Regression Model**

## Task 1: Get Data

You will work with two datasets:

- **Turkish Stock Exchange Data** (`turkish-se-SP500vsMSCI.csv`):
  - **Observations**: 536
  - **Variables**: 1

- **Motor Trend Car Road Tests Data** (`mtcarsdata-4features.csv`):
  - **Observations**: 32
  - **Variables**: 3

These datasets are stored as CSV files. To import them into Python, use the `read_csv()` function from the Pandas library to load them into dataframes.

## Task 2: Fit a Linear Regression Model

You need to compute and visualize linear regression models in the following scenarios:

### 2.1 One-Dimensional Regression Without Intercept

- **Dataset**: Turkish Stock Exchange Data
- **Objective**: Compute linear regression without an intercept.
- **Implementation**: 
  - Use the `BuildGraph()` function.
  - Set the boolean parameter to `TRUE` to compute the regression without an intercept.
  - This involves:
    - Estimating the regression coefficient using `estimate_coefwithout_intercept()`.
    - Computing the Mean Squared Error (MSE) using `OneDimensionalMeanSquareError()`.
    - Plotting the regression line using `plot_regression_line_with_intercept()`.

### 2.2 Regression on Random Subsets

- **Objective**: Compare the linear regression solution on random 10% subsets of the Turkish dataset.
- **Implementation**: 
  - Use the `randomSubset()` function to generate subsets.
  - Visualize results in subplots.

### 2.3 One-Dimensional Regression With Intercept

- **Dataset**: Motor Trend Car Data
- **Objective**: Compute linear regression with an intercept using `mpg` and `weight` columns.
- **Implementation**: 
  - Reuse the `BuildGraph()` function with the boolean parameter set to `FALSE` to include the intercept.

### 2.4 Multi-Dimensional Regression

- **Dataset**: Motor Trend Car Data
- **Objective**: Predict `mpg` using the other three columns (cylinders, horsepower, and weight).
- **Implementation**: 
  - Create and use the `MultiDimensional()` function.
  - Compute the MSE for the multi-dimensional case using `MultiDimensionalMeanSquareError()`.

## Task 3: Test the Regression Model

For this task, you will:

1. **Re-run** Tasks 2.1, 2.3, and 2.4 using only 5% of the data.
2. **Compute the MSE** on both the training data (5%) and testing data (95%).
3. **Repeat** the process for various random splits to validate the robustness of your models.

