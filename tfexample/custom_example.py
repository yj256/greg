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

feature_columns = [physical_fatigue,mental_fatigue, extra_sleep, previous_sleep]

hidden = [3,3]

def my_model_fn(features, labels, mode):
	with tf.name_scope("input"):
		input_layer = tf.feature_column.input_layer(features, feature_columns)
	hidden1 = tf.layers.dense(inputs=input_layer,units=3, activation=tf.nn.relu, name='hidden1')
	hidden2 = tf.layers.dense(inputs=hidden1,units=3, activation=tf.nn.relu, name='hidden2')
	output_layer = tf.layers.dense(inputs=hidden2, units=1, name='output_layer')

	predictions = tf.squeeze(output_layer, 1)

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
		

	loss = tf.losses.mean_squared_error(labels, predictions)
	accuracy = tf.metrics.accuracy(labels,predictions)
	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(
			mode,
			loss=loss,
			eval_metric_ops={'my_accuracy':accuracy})

	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(
		loss, 
		global_step=tf.train.get_global_step(),
		name='train_op_minimize')

	tf.summary.scalar("my_accuracy", accuracy[1])

	if mode == tf.estimator.ModeKeys.TRAIN:
		return tf.estimator.EstimatorSpec(
			mode,
			loss=loss,
			train_op=train_op,)


greg = tf.estimator.Estimator(
	model_fn=lambda features, labels, mode: my_model_fn(features, labels, mode),
	model_dir="./logcus/")


greg.train(input_fn=train_input_fn, steps=2000)

evaluate_result = greg.evaluate(input_fn=test_input_fn, steps=100, name="test")

writer = tf.summary.FileWriter("./logcus")

print("Evaluation results")
for key in evaluate_result:
   print("   {}, was: {}".format(key, evaluate_result[key]))
