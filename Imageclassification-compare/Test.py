from skimage import io, transform
import tensorflow as tf
import numpy as np
#模型读出后 outputs可追加softmax

path1='C:/Users/zzzzz/Desktop/TIM图片20190515001448.jpg'#目标文件路径
W = 32
H= 32
C= 3


def read_one_image(path):
    img = io.imread(path) #图片绝对地址
    img = transform.resize(img, (W,H,C))
    return np.asarray(img)


with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)

    data.append(data1)


    saver = tf.train.import_meta_graph('G:/traffic_data/Model/save_net.meta')#导入模型
    saver.restore(sess, tf.train.latest_checkpoint('G:/traffic_data/Model/'))
    pred= tf.get_collection('network-output')[0]

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0") #https://www.cnblogs.com/hejunlin1992/p/7767912.html,取神经网络最后一次操作
    dropout=graph.get_tensor_by_name('drop:0')
    feed_dict = {x: data,dropout:1}
    classification_result = sess.run(pred, feed_dict)

    # 打印出预测矩阵
    print(classification_result)
    # 打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result, 1).eval())
    # 根据索引通过字典对应标志的分类
