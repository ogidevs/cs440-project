import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def run_apriori(df):
    transactions = []
    mean_price = df['Total_Price_EUR'].mean()
    mean_area = df['Area'].mean()
    
    for _, row in df.iterrows():
        t = [
            f"Advertiser_{row['Advertiser_Type']}",
            "Price_High" if row['Total_Price_EUR'] > mean_price else "Price_Low",
            "Area_Large" if row['Area'] > mean_area else "Area_Small"
        ]
        transactions.append(t)
        
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_array, columns=te.columns_)
    
    freq_items = apriori(df_trans, min_support=0.2, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.6, num_itemsets=2)
    
    return rules[['antecedents', 'consequents', 'support', 'confidence']]