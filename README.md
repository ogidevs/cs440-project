# CS440: Real Estate Data Analytics

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Package Manager: uv](https://img.shields.io/badge/managed_by-uv-purple)](https://github.com/astral-sh/uv)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-blue.svg)](https://scikit-learn.org/)
[![Data Analysis](https://img.shields.io/badge/Analytics-Pandas%20%7C%20NumPy-blue.svg)](https://pandas.pydata.org/)

## Overview

This repository contains the practical project for the **CS440 Data Analytics** course. The objective is to apply end-to-end data science techniques to a real-world dataset:

- **Data cleaning and preprocessing**
- **Exploratory data analysis (EDA)** with visualizations
- **Unsupervised learning** (Clustering with K-Means and DBSCAN, Association Rules with Apriori)
- **Supervised learning** (Classification models: Decision Tree, KNN, Random Forest)
- **Model evaluation and comparison**

### Domain

**Belgrade Real Estate Market** - Analysis of over 8,400 apartment listings with technical characteristics and pricing patterns.

---

## Dataset Overview

| Attribute           | Description                                | Type        | Range                    |
| ------------------- | ------------------------------------------ | ----------- | ------------------------ |
| **Area**            | Apartment size in m²                       | Numeric     | 10 - 638 m²              |
| **Rooms**           | Number of rooms                            | Numeric     | 0.5 - 5 rooms            |
| **Current_Floor**   | Current floor (mapped from Roman numerals) | Numeric     | -1 (SUT) to 30           |
| **Total_Floors**    | Building height                            | Numeric     | 1 - 30 floors            |
| **Total_Price_EUR** | Price in EUR                               | Numeric     | €30K - €5M               |
| **Advertiser_Type** | Seller type                                | Categorical | Agency, Owner, Developer |

**Dataset Statistics:**

- **Total Instances**: 8,446 apartments
- **Original Features**: 12 attributes
- **Final Features**: 9 after preprocessing
- **Target Variable**: `Is_Premium` (Price > €250,000)
- **Data Quality**: Handled missing values, regex extraction, categorical mapping

---

## Key Features & Methodology

### 1. Data Preprocessing & Exploratory Data Analysis (EDA)

#### Data Cleaning Operations

- Removal of redundant columns (ID, Scrape_Date, Photo_Count)
- Regex extraction from text fields (e.g., "3.0 rooms" → 3.0)
- Categorical mapping (Roman numerals to integers: I→1, II→2, ..., SUT→-1, PR→0)
- Missing value imputation using **median** (robust to outliers in real estate market)

#### Exploratory Visualizations Generated

| Visualization                 | Insight                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **price_distribution.png**    | Right-skewed histogram showing price concentration in €100K-€400K zone; luxury outliers at high end           |
| **price_by_advertiser.png**   | Box plot reveals agencies sell the most expensive properties; owners show highest variance (often overpriced) |
| **area_vs_price_scatter.png** | Strong positive correlation (r ≈ 0.85) with trend line: **area is PRIMARY price driver**                      |
| **correlation_matrix.png**    | Heatmap identifies relationships: Area ↔ Price (0.85), Rooms ↔ Area (0.78)                                    |

---

### 2. Clustering (Unsupervised Learning)

#### K-Means Clustering (k=3)

**Purpose**: Market segmentation based on Area and Price

**Results - Three Market Segments**:

- **Cluster 0 (Budget)**: Small apartments (40-60 m²), low price (€30K-€150K)
- **Cluster 1 (Mid-range)**: Medium apartments (60-90 m²), average price (€150K-€350K)
- **Cluster 2 (Premium)**: Large apartments (>90 m²), high price (€350K+)

#### DBSCAN Clustering (eps=0.1957, min_samples=10)

**Purpose**: Density-based anomaly detection (does NOT force data into clusters)

**Results**:

- **Cluster 0 & 1**: 8,095 apartments (95.8%) - Normal market segments
- **Noise (-1)**: 351 apartments (4.2%) - Luxury anomalies & outliers
- **Advantage over K-Means**: Better at detecting market anomalies

**Comparison**:

- K-Means: Better for business segmentation (3 customer groups)
- DBSCAN: Better for anomaly detection (find unusual properties)

---

### 3. Association Rules Mining (Apriori Algorithm)

**Objective**: Discover "If X then Y" market patterns

**Parameters**: min_support=20%, min_confidence=60%

#### Key Market Finding

**Agencies have monopoly on Belgrade real estate market** across ALL property sizes and price ranges. Surprising finding: despite controlling luxury properties, 91% of agency listings are actually budget apartments.

---

### 4. Classification (Supervised Learning)

**Objective**: Predict if apartment is "Premium" (Price > €250,000) based on physical features

#### Features Used

- Area (m²)
- Rooms (count)
- Current_Floor
- Total_Floors

#### Models Tested & Comparison

| Model                         | Accuracy   | Precision  | Recall     | F1-Score   | Status   |
| ----------------------------- | ---------- | ---------- | ---------- | ---------- | -------- |
| Decision Tree                 | 83.14%     | 0.7955     | 0.8184     | 0.8068     | Baseline |
| K-Nearest Neighbors (k=5)     | 85.68%     | 0.8336     | 0.8336     | 0.8336     | Good     |
| **Random Forest** (100 trees) | **86.63%** | **0.8417** | **0.8487** | **0.8452** | BEST     |

#### Best Model: Random Forest

**Why Random Forest wins**:

1. **Bagging**: Reduces variance by training on different data subsets
2. **Ensemble strength**: 100 trees aggregate predictions; individual errors cancel out
3. **Outier robustness**: Less sensitive to luxury property anomalies
4. **Non-linear relationships**: Captures complex interactions between features
5. **Feature importance**: Automatically ranks which features matter most

**Training Configuration**:

- **Train/Test Split**: 80% / 20% (6,756 train, 1,690 test samples)
- **Stratification**: Maintains Premium class balance in both sets
- **Scaling**: StandardScaler applied to normalize features

---

## Project Architecture

```
cs440-real-estate-analytics/
│
├── data/
│   └── dataset.csv                      # Raw data: 8,446 apartments
│
├── outputs/                             # Auto-generated visualizations
│   ├── price_distribution.png           # Histogram of prices
│   ├── price_by_advertiser.png          # Box plot: Advertisers vs Prices
│   ├── area_vs_price_scatter.png        # Scatter + trend line
│   ├── correlation_matrix.png           # Heatmap of feature correlations
│   ├── clustering_results.png           # K-Means vs DBSCAN visualization
│   └── k_distance_graph.png             # K-distance for DBSCAN tuning
│
├── src/                                 # Source code modules
│   ├── __init__.py
│   ├── data_processing.py               # Data cleaning, regex, imputation
│   ├── eda.py                           # EDA visualizations (4 plots)
│   ├── clustering.py                    # K-Means + DBSCAN with diagnostics
│   ├── association.py                   # Apriori algorithm
│   └── classification.py                # 3 ML models with metrics
│
├── main.py                              # Main execution pipeline
├── pyproject.toml                       # Project dependencies & metadata
├── uv.lock                              # Locked dependency versions
├── README.md                            # Documentation
```

---

## Installation & Setup

### Prerequisites

- **Python 3.13+**
- **uv** package manager ([install here](https://docs.astral.sh/uv/))

### Quick Start

```bash
# Install dependencies (creates virtual environment)
uv sync
```

---

## Usage

### Run Complete Analysis Pipeline

```bash
uv run python main.py
```

**Output includes**:

1. Data cleaning report with shape info
2. Descriptive statistics (mean, median, std)
3. EDA visualizations (saved to `outputs/` folder)
4. Clustering diagnostics (number of clusters, noise points)
5. Top 5 association rules with support/confidence
6. Model comparison table (Accuracy, Precision, Recall, F1)
7. Best model recommendation & detailed analysis

---

## Key Findings & Insights

### Main Discoveries

1. **Market Monopoly**: Agencies control **90%+** of listings across ALL segments
2. **Price Driver**: A property's **area (m²)** explains ~85% of price variance (strongest predictor)
3. **Outlier Detection**: DBSCAN identified ~4.2% of properties as luxury anomalies
4. **Market Segmentation**: Natural market splits into 2-3 distinct price clusters
5. **Premium Prediction**: Random Forest achieves **86.6% accuracy** predicting Premium properties
6. **Owner Premium**: Owners rarely sell premium properties; agencies dominate that segment

---

## Conclusions & Future Improvements

### Project Achievements

- Complete end-to-end ML pipeline implemented
- Market segmentation achieved (K-Means: 3 clusters)
- Anomaly detection successful (DBSCAN: 4.2% outliers)
- Association rule mining revealed market monopoly
- Premium property prediction at 86.6% accuracy
- Professional code structure with modular design

### Recommended Future Enhancements

1. **Geospatial Analysis**
   - Add latitude/longitude features
   - Implement location-based pricing models
   - Distance-to-center impact on prices

2. **Natural Language Processing**
   - Parse apartment descriptions
   - Extract amenities (balcony, garden, renovated, etc.)
   - Sentiment analysis on advertiser tone

3. **Temporal Analysis**
   - Track price trends over time
   - Seasonal patterns (more/fewer listings)
   - Market boom/bust cycles

4. **Advanced ML**
   - Hyperparameter tuning (GridSearchCV)
   - Neural networks for complex patterns
   - XGBoost for better accuracy

5. **Production Deployment**
   - Flask/FastAPI REST API
   - Real-time price predictions
   - Model serving with Docker
