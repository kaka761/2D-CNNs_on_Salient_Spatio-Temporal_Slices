import pandas as pd
#dic_sorted = sorted(dic.items(), key = lambda x:x[1], reverse = True)
df = pd.read_csv('./Projects/slice_ar/csv/ipn/test.csv')
df = df.sort_values('idx')
df.to_csv('./Projects/slice_ar/csv/ipn/test_sort.csv', index = False)