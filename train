import tensorflow as tf
import model
import time

w=128
h=128
c=3
num_class = 2

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label


img, label = read_and_decode("train.tfrecords")

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=64, capacity=2000,
                                                min_after_dequeue=1000)


x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x') #tensor
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = model.inference(x,regularizer)

#(小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval')

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver=tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()  # 开启多线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 这里指代的是读取数据的线程，如果不加的话队列一直挂起

    for epoch in range(100):
        tra_acc = 0
        for i in range(1,1000):
            start_time = time.time()

            x_train_a,y_train_a = sess.run([img_batch,label_batch])
            sess.run(train_op,feed_dict={x: x_train_a, y_: y_train_a})

            train_accuracy = sess.run(acc, feed_dict={x: x_train_a, y_: y_train_a})
            train_loss = sess.run(loss, feed_dict={x: x_train_a, y_: y_train_a})
            duration = time.time() - start_time
            if i % 10 == 0:
                print("Iter:%d , train_acc:%.2f" %(i,train_accuracy))

                saver.save(sess,'E:\\tensorflow\\tfrecord\\logs\\model.ckpt')
    coord.request_stop()  # 多线程关闭
    coord.join(threads)

