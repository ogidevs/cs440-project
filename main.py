from src.data_processing import load_and_clean_data
from src.eda import perform_eda
from src.clustering import run_clustering
from src.association import run_apriori
from src.classification import run_classification

def main():
    print("1. Loading and cleaning data...")
    df = load_and_clean_data('data/dataset.csv')
    print(f"Data shape after cleaning: {df.shape}")
    
    print("\n2. Running Exploratory Data Analysis (EDA)...")
    perform_eda(df)
    print("EDA completed. Plots saved to 'outputs/' directory.")
    
    print("\n3. Running Clustering (K-Means & DBSCAN)...")
    df = run_clustering(df)
    print("Clustering completed. Plots saved.")
    
    print("\n4. Running Association Rules (Apriori)...")
    rules = run_apriori(df)
    print("Top 5 Association Rules found:")
    print(rules.head().to_string())
    
    print("\n5. Running Classification Models...")
    ml_results = run_classification(df)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON - PERFORMANCE METRICS")
    print("="*60)
    
    for model_name, metrics in ml_results.items():
        print(f"\n--- {model_name} ---")
        print(f"Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall:    {metrics['Recall']:.4f}")
        print(f"F1-Score:  {metrics['F1_Score']:.4f}")
        print(f"\nClassification Report:\n{metrics['Report']}")
    
    # Analyze and recommend best model
    print("\n" + "="*60)
    print("MODEL ANALYSIS & RECOMMENDATION")
    print("="*60)
    
    best_f1 = max(ml_results.items(), key=lambda x: x[1]['F1_Score'])
    
    print(f"\n[BEST MODEL - F1-Score] {best_f1[0]}")
    print(f"   F1-Score: {best_f1[1]['F1_Score']:.4f}")
    print(f"   Accuracy: {best_f1[1]['Accuracy']:.4f}")
    print(f"   Precision: {best_f1[1]['Precision']:.4f}")
    print(f"   Recall: {best_f1[1]['Recall']:.4f}")

if __name__ == "__main__":
    main()