import pandas as pd 
import numpy as np
import sys
import math

lines = int(sys.argv[1])
path = sys.argv[2]
cols = ['regular_exercise','acute_4_before', 'acute_8_before', 'acute_4_to_8','previous_sleep', 'target']
df = pd.DataFrame([[0,0,0,0,8,7]],columns=cols)


for x in range(1,lines):
	#print(df)
	#print(x)
	prev = df.iloc[x-1]['target']
	reg = np.random.randint(0,24)
	a4 = np.random.randint(0,5)
	a8 = np.random.randint(0,17)
	a48= np.random.randint(0,5)
	t = 5+ np.sin(reg*np.pi/46) + np.exp(a4)/(np.log(a8+1) * np.exp(a48) + 2) + prev/(prev+2)
	df = df.append(pd.DataFrame([[reg,a4,a8,a48,prev,t]],columns=cols))
df.to_csv(path_or_buf=path)