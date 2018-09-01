import tensorflow as tf

train_path = '/home/nick/Downloads/low_freq/house_1/channel_1.dat'
train_path2 = '/home/nick/Downloads/low_freq/house_1/channel_2.dat'


def _parse_line(line):
    st = tf.string_split([line], " ")
    timestamp = tf.strings.to_number(st.values[0], tf.int64)
    value = tf.strings.to_number(st.values[1])
    #return tf.convert_to_tensor([timestamp, value])
    return {"Timestamp": timestamp, "Value": value}


# def _process_dataset(tensor1, tensor2):
#     timestamp = tensor1[0]
#     value1 = tensor1[1]
#     value2 = tensor2[1]
#     return tf.convert_to_tensor([timestamp,value1,value2])

def _process_dataset(tensor1, tensor2):
    timestamp = tensor1['Timestamp']
    value1 = tensor1['Value']
    value2 = tensor2['Value']
    return {"Timestamp": timestamp, "Values": tf.convert_to_tensor([value1,value2])}


ds = tf.data.TextLineDataset(train_path)#.skip(1)
ds = ds.map(lambda x: _parse_line(x))

ds2 = tf.data.TextLineDataset(train_path2)#.skip(1)
ds2 = ds2.map(lambda x: _parse_line(x))

ds3 = tf.data.Dataset.zip((ds, ds2))
ds3 = ds3.map(lambda x, y: _process_dataset(x, y))
print(ds3)

iterator = ds3.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
        value = sess.run(next_element)
        print(value)