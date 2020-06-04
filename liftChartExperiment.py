# %%
# packages and data
import pandas as pd
import numpy as np
# PredDF = my_ys
PredDF = pd.read_csv("/Users/jhalstead/Documents/data/project1Out/my_ys.csv")
PredDF = PredDF.drop('Unnamed: 0', axis='columns')

my_names = ["Random Forest", "Cloned R RF", "AdaBoost", "Soft Voter"]
LiftDF = []

# loop
for model1 in my_names:
    print(model1)
    df1 = PredDF[['actual', model1]]
    df1['decile'] = pd.qcut(df1[model1], 10, labels=False)
    df1.columns = ['customer', 'probability', 'decile']
    df2 = pd.pivot_table(data=df1, index=['decile'], values=['customer', 'probability'],
                     aggfunc={'customer': [np.sum], 'probability': [np.min,np.max]})
    df2.reset_index()
    df2.columns = ['customer_count', 'max_score', 'min_score']
    df3 = df2.sort_values(by='min_score', ascending=False)
    df3['gain'] = np.round(((df3['customer_count'] / df3['customer_count'].sum()).cumsum()), 4) * 100
    df3['bucket'] = np.arange(10, 101, 10).tolist()
    df3['lift'] = (df3['gain']/df3['bucket'])
    df3['group'] = list(range(1, 11))
    df3['model'] = model1
    df4 = df3[['group', 'lift', 'model']].copy()
    LiftDF.append(df4)

print(LiftDF)
