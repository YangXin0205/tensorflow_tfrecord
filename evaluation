
from skimage import io,transform
import tensorflow as tf
import numpy as np


path1 = r"E:\data\dataset\test_image\6.jpg"
path2 = r"E:\data\dataset\test_image\7.jpg"
path3 = r"E:\data\dataset\test_image\8.jpg"
path4 = r"E:\data\dataset\test_image\3.jpg"
path5 = r"E:\data\dataset\test_image\11.jpg"

flower_dict = {0:'man',1:'woman'}

w=128
h=128
c=3

def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)

with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)
    data2 = read_one_image(path2)
    data3 = read_one_image(path3)
    data4 = read_one_image(path4)
    data5 = read_one_image(path5)
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)

    saver = tf.train.import_meta_graph('E:\\tensorflow\\cnn_classifier\\logs\\model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('E:\\tensorflow\\cnn_classifier\\logs\\'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)

    #打印出预测矩阵
    print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result,1).eval())
    #根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("第",i+1,"个人性别为:"+flower_dict[output[i]])
