import sys
import os
sys.path.append('../')
import pandas as pd
from config import OUT_EVAL
df = pd.read_csv(OUT_EVAL + 'last/result.csv')


missed = df['missed detection'].sum()
false_alarm = df['false alarm'].sum()
total = df['total'].sum()
correct = df['correct'].sum()
confusion = df['confusion'].sum()
der = (false_alarm + missed + confusion)/total
per = correct/total
print(df)
print()
print()
# print(df.loc[df['filename'] == '4065.cha'])
print('DER na całym zbiorze CALLHOME ', der)
# print(per)
# print(per + der)
print('missed / total', missed/ total)
print('false_alarm / total', false_alarm/ total)
print('correct / (correct + confusion)', correct/ (correct + confusion))
