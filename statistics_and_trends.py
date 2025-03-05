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
    """Plots a scatter plot showing Age vs. Time Spent on Social Media."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df["age"], y=df["time_spent"], hue=df["platform"], alpha=0.7)
    plt.title("Age vs. Time Spent on Social Media")
    plt.xlabel("Age")
    plt.ylabel("Time Spent (hours/mins)")
    plt.legend(title="Platform")
    plt.savefig('relational_plot.png')
    plt.show()

def plot_categorical_plot(df):
    """Plots a bar chart showing the average time spent per platform."""
    plt.figure(figsize=(8, 5))
    sns.barplot(x=df["platform"], y=df["time_spent"], hue=df["platform"], estimator=np.mean, palette="coolwarm", dodge=False)
    plt.title("Average Time Spent on Social Media by Platform")
    plt.xlabel("Social Media Platform")
    plt.ylabel("Average Time Spent")
    plt.xticks(rotation=45)
    plt.savefig('categorical_plot.png')
    plt.show()

def plot_statistical_plot(df):
    """Plots a heatmap showing correlations between numerical features."""
    plt.figure(figsize=(8, 5))
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numerical Variables")
    plt.savefig('statistical_plot.png')
    plt.show()

def statistical_analysis(df, col: str):
    """Computes statistical moments for a given numerical column."""
    mean = np.mean(df[col])
    stddev = np.std(df[col], ddof=1)
    skew = df[col].skew()
    excess_kurtosis = df[col].kurt()
    return mean, stddev, skew, excess_kurtosis

def preprocessing(df):
    """Preprocesses the dataset by handling missing values and summarizing data."""
    df.dropna(inplace=True)
    print("\nData Summary:\n", df.describe())
    print("\nFirst Five Rows:\n", df.head())
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    print("\nCorrelation Matrix:\n", numeric_df.corr())  # Only numeric columns
    return df

def writing(moments, col):
    """Prints the statistical moments analysis for the selected column."""
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    if abs(moments[2]) < 2:
        skew_text = "not skewed"
    elif moments[2] > 2:
        skew_text = "right skewed"
    else:
        skew_text = "left skewed"
    
    if moments[3] < -1:
        kurtosis_text = "platykurtic"
    elif -1 <= moments[3] <= 1:
        kurtosis_text = "mesokurtic"
    else:
        kurtosis_text = "leptokurtic"
    
    print(f'The data was {skew_text} and {kurtosis_text}.')
    return

def main():
    """Main function to load data, process it, generate plots, and perform analysis."""
    df = pd.read_csv('dummy_data.csv')
    df = preprocessing(df)
    col = 'time_spent'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return

if __name__ == '__main__':
    main()
