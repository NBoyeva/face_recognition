import pandas as pd

df = pd.read_csv('dataframe.csv')
print(df)

df2 = pd.DataFrame([['0', '1']], columns=['name','image'])
df = pd.concat([df,df2], axis=0)
print(df)
df.to_csv('dataframe.csv', index=False)
