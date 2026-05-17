from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

def run_classification(df):
    # Target: 1 if Price > 250k EUR
    df['Is_Premium'] = (df['Total_Price_EUR'] > 250000).astype(int)
    
    X = df[['Area', 'Rooms', 'Current_Floor', 'Total_Floors']]
    y = df['Is_Premium']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        report = classification_report(y_test, preds)
        results[name] = {
            "Accuracy": acc, 
            "F1_Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Report": report
        }
        
    return results