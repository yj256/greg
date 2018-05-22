import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import sys

log_path = sys.argv[1]
b_size = int(sys.argv[2])
PATH = sys.argv[3]




tf.logging.set_verbosity(tf.logging.FATAL)

# loading the data
train_path = sys.argv[4]	#training data path
test_path = "./test_182.csv"	#testing data path
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

train_input_fn = tf.estimator.inputs.pandas_input_fn(
					x=train_data[['regular_exercise','acute_4_before', 'acute_8_before', 'acute_4_to_8','previous_sleep']],
					y=train_data[['target']],
					batch_size=b_size,
					num_epochs=None,
					shuffle=True,
					)

test_input_fn = tf.estimator.inputs.pandas_input_fn(
					x=test_data[['regular_exercise','acute_4_before', 'acute_8_before', 'acute_4_to_8','previous_sleep']],
					y=test_data[['target']],
					batch_size=30,
					num_epochs=1,
					shuffle=True,
					)



# Define four numeric feature columns.
regular_exercise = tf.feature_column.numeric_column('regular_exercise')
acute_4_before = tf.feature_column.numeric_column('acute_4_before')
acute_8_before = tf.feature_column.numeric_column('acute_8_before')
acute_4_to_8 = tf.feature_column.numeric_column('acute_4_to_8')
previous_sleep = tf.feature_column.numeric_column("previous_sleep")

feature_cols = [regular_exercise,acute_4_before, acute_8_before, acute_4_to_8,previous_sleep]

#hidden layer(s) layout
hidden = [3,3]

#measurement of root mean square error
def my_rmse(labels, predictions):
	labels = tf.cast(labels, tf.float64)
	pred_values = predictions['predictions']
	pred_values = tf.cast(pred_values, tf.float64)
	return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)}

#Instantiate an estimator, passing the feature columns.
greg = tf.estimator.DNNRegressor(
	feature_columns=feature_cols,
	hidden_units=hidden,
	activation_fn=tf.nn.relu,
	model_dir=log_path,
	optimizer=tf.train.AdamOptimizer()
	)


#greg = tf.contrib.estimator.add_metrics(greg, my_accuracy)
greg = tf.contrib.estimator.add_metrics(greg, my_rmse)

df = pd.DataFrame([[0,0,0,0,0]],columns=['batch_size','average_loss','loss','global_step','rmse'])
steps = [1000,4000,5000,10000,10000,10000,10000,10000,10000,10000,10000,10000]
for i in steps:
	greg.train(input_fn=train_input_fn, steps=i)
	evaluate_result = greg.evaluate(input_fn=test_input_fn, steps=300, name="test")
	print("Evaluation results")
	arr= np.array([b_size])
	for key in evaluate_result:
		arr = np.append(arr,[evaluate_result[key]])
		print("   {}, was: {}".format(key, evaluate_result[key]))
	df = df.append(pd.DataFrame([arr],columns=['batch_size','average_loss','loss','global_step','rmse']))
df.to_csv(path_or_buf=PATH)