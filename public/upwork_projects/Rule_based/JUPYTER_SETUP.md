# Jupyter Notebook Setup Instructions

## How to Fix Graph Display Issues

### 1. Enable Matplotlib Inline Display
At the very beginning of your Jupyter notebook (first cell), add this magic command:

```python
%matplotlib inline
```

This tells Jupyter to display matplotlib plots inline instead of trying to show them in separate windows.

### 2. About R² Scores
The R² scores appear negative because the ML models perform worse than simply predicting the mean revenue. This is common with:

- Small datasets (like our 5,000 transactions)
- Limited predictive features
- High variance in the target variable

**Solution**: Focus on MSE and RMSE metrics instead, which are more appropriate for regression evaluation in this context.

### 3. Expected Behavior
After adding `%matplotlib inline`:
- All graphs will display directly in the notebook cells
- No separate plot windows will open
- Charts will be embedded in the output cells

### 4. Alternative: Interactive Plots
If you want interactive plots, you can use:
```python
%matplotlib widget
```

But `%matplotlib inline` is usually sufficient for most use cases.

## Quick Fix for Your Notebook

1. Open your `rulebased.ipynb` file
2. Add a new cell at the very top
3. Put this code in it: `%matplotlib inline`
4. Run all cells - graphs should now display properly!

## Why This Happens

The original code was designed to run as a Python script, which saves plots to PNG files. Jupyter notebooks have different display requirements, hence the need for the `%matplotlib inline` magic command.