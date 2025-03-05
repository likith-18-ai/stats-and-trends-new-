"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

def plot_relational_plot(df):
    """Generates and saves a relational plot."""
    sns.pairplot(df)
    plt.savefig('relational_plot.png')
    plt.close()

def plot_categorical_plot(df):
    """Generates and saves a categorical plot."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.savefig('categorical_plot.png')
    plt.close()

def plot_statistical_plot(df):
    """Generates and saves a statistical plot."""
    plt.figure(figsize=(10, 6))
    for column in df.select_dtypes(include=np.number).columns:
        sns.histplot(df[column], kde=True, label=column)
    plt.legend()
    plt.savefig('statistical_plot.png')
    plt.close()

def statistical_analysis(df, col: str):
    """Computes mean, standard deviation, skewness, and excess kurtosis for a given column."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col], nan_policy='omit')
    excess_kurtosis = ss.kurtosis(df[col], nan_policy='omit')
    return mean, stddev, skew, excess_kurtosis

def preprocessing(df):
    """Performs basic data preprocessing and prints summary statistics."""
    print("Data Summary:\n", df.describe())
    print("\nCorrelation Matrix:\n", df.corr())
    print("\nFirst few rows:\n", df.head())
    return df

def writing(moments, col):
    """Prints the statistical analysis results."""
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    if moments[2] > 2 or moments[2] < -2:
        skewness = "skewed"
    else:
        skewness = "not skewed"
    kurtosis_type = "mesokurtic"
    if moments[3] > 2:
        kurtosis_type = "leptokurtic"
    elif moments[3] < -2:
        kurtosis_type = "platykurtic"
    print(f'The data was {skewness} and {kurtosis_type}.')

def main():
    """Main function to execute the analysis pipeline."""
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = df.select_dtypes(include=np.number).columns[0]
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)

if __name__ == '__main__':
    main()
