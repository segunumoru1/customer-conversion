import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"No such file: '{file_path}'") 
    
    # dataset info and description
def dataset_info(df):
    print("Dataset Information:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nMissing Values:")
    print(df.isnull().sum())

# correlation matrix
def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    selected_columns = df.select_dtypes(include=[np.number]).columns[1:4].tolist()  # Select only numerical columns
    corr = df[selected_columns].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.savefig('../artifacts/correlation_matrix.png')
    plt.close()

# feature distribution
def plot_feature_distributions(df):
    numerical_features = df.select_dtypes(include=[np.number]).columns[1:4].tolist()  # Exclude 'id' and 'target' columns if they exist
    fig, axes = plt.subplots(nrows=1, ncols=len(numerical_features), figsize=(20, 5))
    for i, feature in enumerate(numerical_features):
        sns.histplot(df[feature], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('../artifacts/feature_distributions.png')
    plt.close()

if __name__ == "__main__":
    # Example usage
    file_path = "C:\\Users\\SEGUN\\customer-conversion\\data\\week2_marketing_data.csv"  # Replace with your actual file path
    try:
        df = load_data(file_path)
        dataset_info(df)
        plot_correlation_matrix(df)
        plot_feature_distributions(df)
    except FileNotFoundError as e:
        print(e)
