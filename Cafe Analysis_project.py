import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# loading csv file
mydata = pd.read_csv('bread basket.csv')

# teacher said to clean data so doing this
mydata['Item'] = mydata['Item'].str.lower()
mydata['Item'] = mydata['Item'].str.strip()

# removing adjustment i dont know what this is but removing it
mydata = mydata[mydata['Item'] != 'adjustment']

# just checking
print(mydata.head())
print(len(mydata))

# top 10 items
top10items = mydata['Item'].value_counts().head(10)

# making graph
plt.figure(figsize=(10,5))
sns.barplot(x=top10items.values, y=top10items.index, palette='coolwarm')
plt.title('top 10 items')
plt.xlabel('count')
plt.ylabel('items')
plt.show()

# grouping by transaction
# basically all items bought together
alltransactions = mydata.groupby('Transaction')['Item'].apply(list).values.tolist()

# this converts it to matrix
# copied from stackoverflow
myencoder = TransactionEncoder()
encodeddata = myencoder.fit(alltransactions).transform(alltransactions)
finalmatrix = pd.DataFrame(encodeddata, columns=myencoder.columns_)

# apriori
myitemsets = apriori(finalmatrix, min_support=0.01, use_colnames=True)

# rules
myrules = association_rules(myitemsets, metric="lift", min_threshold=1.0)
myrules = myrules.sort_values('lift', ascending=False).reset_index(drop=True)

print(myrules[['antecedents','consequents','support','confidence','lift']].head(10))