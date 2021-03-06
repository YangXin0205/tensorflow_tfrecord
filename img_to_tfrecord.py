import os
import tensorflow as tf
from PIL import Image

cwd = "E:\\data\\dataset\\man_woman\\"

classes = ['man','woman']

writer = tf.python_io.TFRecordWriter("train.tfrecords")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()  # 开启多线程
    threads = tf.train.start_queue_runners(coord=coord)
    for index, name in enumerate(classes):
        class_path = cwd + name + "\\"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((128, 128))
            img_raw = img.tobytes()              #将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())  #序列化为字符串
    coord.request_stop()  # 多线程关闭
    coord.join(threads)
    writer.close()
