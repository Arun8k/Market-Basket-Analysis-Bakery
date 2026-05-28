import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import os
# ==========================================
# 1. LOAD AND CLEAN DATA
# ==========================================
# Look for the dataset locally first, and fall back to the absolute desktop path
candidate_paths = [
    'bread basket.csv',
    'bread\\ basket.csv',
    '/Users/cashify/Desktop/IITG_Market_Basket_Project/bread basket.csv',
    '/Users/cashify/Desktop/IITG_Market_Basket_Project/bread\\ basket.csv'
]
cleaned_path = None
for path in candidate_paths:
    p = path.replace('\\', '')
    if os.path.exists(p):
        cleaned_path = p
        break
if not cleaned_path:
    cleaned_path = '/Users/cashify/Desktop/IITG_Market_Basket_Project/bread basket.csv'
print(f"Loading dataset from: {cleaned_path}")
df = pd.read_csv(cleaned_path)
# Clean text formatting: convert to lowercase and strip white spaces
df['Item'] = df['Item'].str.strip().str.lower()
# Drop rows where Item is 'adjustment' (backend operations)
df = df[df['Item'] != 'adjustment']
print("--- DATASET PREVIEW ---")
print(df.head())
print(f"\nTotal rows loaded: {len(df)}")
# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
# Compute top 10 products sold
top_items = df['Item'].value_counts().head(10)
# Plot the bar chart
plt.figure(figsize=(10, 5))
sns.barplot(x=top_items.values, y=top_items.index, palette='coolwarm')
plt.title('Top 10 Most Frequently Purchased Items')
plt.xlabel('Total Transaction Count')
plt.ylabel('Product Name')
plt.tight_layout()
plt.show()
# ==========================================
# 3. TRANSFORMATION & MATRIX SETUP
# ==========================================
# Group individual transactions into lists based on Transaction ID
transactions = df.groupby('Transaction')['Item'].apply(list).values.tolist()
# One-hot encode item lists into a boolean matrix table
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
transaction_matrix = pd.DataFrame(te_ary, columns=te.columns_)
print("\n--- ONE-HOT MATRIX SHAPE ---")
print(f"Rows (Transactions): {transaction_matrix.shape[0]}")
print(f"Columns (Unique Items): {transaction_matrix.shape[1]}")
# ==========================================
# 4. MODELING (APRIORI ALGORITHM)
# ==========================================
# Mine frequent itemsets with a minimum support threshold of 1% (0.01)
frequent_itemsets = apriori(transaction_matrix, min_support=0.01, use_colnames=True)
# Generate predictive association rules where Lift >= 1.0
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)
# Display the top 10 strongest business rules found
print("\n--- TOP 10 EXTRACTED ASSOCIATION RULES ---")
if not rules.empty:
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
else:
    print("No rules found meeting the minimum thresholds. Try lowering min_support.")
# ==========================================
# 5. RULE DISTRIBUTION SCATTER PLOT
# ==========================================
if not rules.empty:
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='YlOrRd', alpha=0.8, edgecolors='black')
    plt.colorbar(scatter, label='Lift Score')
    plt.title('Association Rules Distribution: Support vs. Confidence')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
