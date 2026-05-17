import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Price Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Total_Price_EUR'], bins=40, kde=True, color='blue')
    plt.title('Distribution of Property Prices')
    plt.savefig(f"{output_dir}/price_distribution.png")
    plt.close()
    
    # 2. Boxplot: Price by Advertiser
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Advertiser_Type', y='Total_Price_EUR', data=df)
    plt.title('Price by Advertiser Type')
    plt.savefig(f"{output_dir}/price_by_advertiser.png")
    plt.close()
    
    # 3. Correlation Heatmap (Expanded feature)
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    # 4. Descriptive Statistics
    print("\n=== Descriptive Statistics ===")
    print(df[['Area', 'Rooms', 'Total_Price_EUR', 'Current_Floor', 'Total_Floors']].describe())

    # 5. Scatter Plot: Area vs Price with trend line
    plt.figure(figsize=(8, 5))
    sns.regplot(x='Area', y='Total_Price_EUR', data=df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    plt.title('Area vs Price')
    plt.savefig(f"{output_dir}/area_vs_price_scatter.png")
    plt.close()