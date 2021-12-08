import numpy as np
import tensorflow as tf
def integration_task(seq_len, num_sapmles):
    curr_num_samples = 0
    while curr_num_samples < num_sapmles:
        rand_vec = np.random.normal(size=seq_len)
        label = int(np.sum(rand_vec, axis=-1) >= 0)
        rand_vec = np.expand_dims(rand_vec, -1)
        yield rand_vec, label
        curr_num_samples += 1

def my_integration_task():
    my_seq_len = 25
    my_num_samples = 10
    for item in integration_task(my_seq_len, my_num_samples):
        yield item


for elem in my_integration_task():
    print(elem)

shapes = ((25, 1), (1))
dataset = tf.data.Dataset.from_generator(generator=my_integration_task,output_types=(tf.float32, tf.int32),output_shapes=shapes)

print(dataset.take(20))
#it = iter(dataset)

#print(next(it))

#for myelem in dataset.take(2):
#    print(myelem)