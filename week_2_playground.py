import numpy as np
import tensorflow as tf

# input: numbers from 0 to 10
a = np.arange(0, 10)
print('input numpy.array', a)

# dataset from input:
dataset = tf.data.Dataset.from_tensor_slices(a)

def nested_print(message, ds):
	print(message)
	for w in ds:
		for x in w:
			print(x, end=",")
		print("\n")
		

# window dataset with shift 1:
window_size = 1
window_dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
nested_print('printing the window dataset:', window_dataset)
		
batch_dataset_1 = window_dataset.flat_map(lambda window: window.batch(window_size + 1))
print("\n")
nested_print('printing the batch_dataset_1:', batch_dataset_1)

shuffle_buffer_size = 9
shuffled_dataset = batch_dataset_1.shuffle(shuffle_buffer_size).map(lambda window: (window[:-1], window[-1]))
nested_print('printing the shuffled dataset:', shuffled_dataset)
		
batch_size = 3
batch_dataset_2 = dataset.batch(batch_size).prefetch(1)
nested_print('printing the batch_dataset_2:', batch_dataset_2)
