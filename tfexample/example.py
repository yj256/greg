import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 


tf.logging.set_verbosity(tf.logging.FATAL)


# loading the data
train_path = "./train.csv"
test_path = "./test.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

train_input_fn = tf.estimator.inputs.pandas_input_fn(
					x=train_data[['physical','mental', 'extra', 'previous']],
					y=train_data[['target']],
					batch_size=2000,
					num_epochs=None,
					shuffle=True,
					)

test_input_fn = tf.estimator.inputs.pandas_input_fn(
					x=test_data[['physical','mental', 'extra', 'previous']],
					y=test_data[['target']],
					batch_size=200,
					num_epochs=1,
					shuffle=True,
					)



# Define four numeric feature columns.
physical_fatigue = tf.feature_column.numeric_column('physical')
mental_fatigue = tf.feature_column.numeric_column('mental')
extra_sleep = tf.feature_column.numeric_column('extra')
previous_sleep = tf.feature_column.numeric_column('previous')

feature_cols = [physical_fatigue,mental_fatigue, extra_sleep, previous_sleep]

hidden = [3,3]

def my_rmse(labels, predictions):
	labels = tf.cast(labels, tf.float64)
	pred_values = predictions['predictions']
	pred_values = tf.cast(pred_values, tf.float64)
	return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)}
def my_accuracy(labels, predictions):
	labels = tf.cast(labels, tf.float64)
	pred_values = predictions['predictions']
	pred_values = tf.cast(pred_values, tf.float64)
	return {'accuracy': tf.metrics.accuracy(labels, pred_values)}

#Instantiate an estimator, passing the feature columns.
greg = tf.estimator.DNNRegressor(
	feature_columns=feature_cols,
	hidden_units=hidden,
	activation_fn=tf.nn.relu,
	model_dir="./logs/",
	optimizer=tf.train.AdamOptimizer()
	)

greg = tf.contrib.estimator.add_metrics(greg, my_rmse)
greg = tf.contrib.estimator.add_metrics(greg, my_accuracy)
##accuracy = tf.metrics,accuracy()
greg.train(input_fn=train_input_fn, steps=30)

'''
with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

sum_saver = tf.train.SummarySaverHook(save_steps=10, output_dir="/Documents/GREG/tfexample",
									summary_op=acc)
'''
evaluate_result = greg.evaluate(input_fn=test_input_fn, steps=200, name="test")


print("Evaluation results")
for key in evaluate_result:
   print("   {}, was: {}".format(key, evaluate_result[key]))

'''
for key in greg.get_variable_names():
	print("{} has weight {}".format(key, greg.get_variable_value(key)))
'''