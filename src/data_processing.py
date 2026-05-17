import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, on_bad_lines='skip')
    
    # Drop irrelevant columns
    cols_to_drop = ['ID', 'Scrape_Date', 'Photo_Count']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Extract numbers from text columns
    df['Rooms'] = df['Rooms'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    df['Total_Floors'] = df['Total_Floors'].astype(str).str.extract(r'(\d+)').astype(float)
    
    # Map Roman numerals to int
    floor_mapping = {
        'SUT': -1, 'PSUT': -0.5, 'PR': 0, 'VPR': 0.5,
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 
        'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13, 
        'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18, 
        'XIX': 19, 'XX': 20, 'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24,
        'XXV': 25, 'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29, 'XXX': 30
    }
    df['Current_Floor'] = df['Current_Floor'].map(floor_mapping)
    
    # Handle NaNs
    df['Rooms'] = df['Rooms'].fillna(df['Rooms'].median())
    df['Current_Floor'] = df['Current_Floor'].fillna(df['Current_Floor'].median())
    df['Total_Floors'] = df['Total_Floors'].fillna(df['Total_Floors'].median())
    df = df.dropna()
    
    return df