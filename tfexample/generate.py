import pandas as pd 
import numpy as np
import sys

lines = int(sys.argv[1])
path = sys.argv[2]
cols = ['physical','mental','extra','previous','target']
df = pd.DataFrame([[0,0,0,9,7.5]],columns=cols)


for x in range(1,lines):
	print(df)
	print(x)
	d = df.iloc[x-1]['target']
	a = np.random.randint(0,11)
	b = np.random.randint(0,11)
	c = np.random.random()
	t = 7.5 + 0.1*a + np.sin(b*np.pi/20) - c/2 + (9-d)/3
	df = df.append(pd.DataFrame([[a,b,c,d,t]],columns=cols))
df.to_csv(path_or_buf=path)