from collections import Counter
import pandas as pd

path = '/home/yhu/Projects/slice_ar/csv/ipn/trainf.csv'
df = pd.read_csv(path)
data = df.values
c = Counter(data[:,2])
print(c)