import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf

#地址：采用绝对路径
root_path ='G:/traffic_data/Training/'

model_path='G:/traffic_data/Model/' #模型存放路径

#---------------------------------------------------------
#超参数设定：
W =32   #图片长款通道数
H =32
C=3
learning_rate=0.001
decay_rate=0.99
n_epoch = 200
batch_size =128
#----------------------------------------------------------
#图片的输入，图片存放于images列表，标签存放于labels列表，images[a]对应目标为labels[b]
directories = [d for d in os.listdir(root_path)
               if os.path.isdir(os.path.join(root_path, d))] #获取标签名
print(directories)
def read_img(root_path):
    labels=[]
    images=[]
    directories = [d for d in os.listdir(root_path)
                   if os.path.isdir(os.path.join(root_path, d))]
    for d in directories:
        label_dir = os.path.join('G:/traffic_data/Training', d)
        # file_names存放所有照片地址
        file_names = [os.path.join(label_dir, f)   # file_names与directorues列表生成器类比理解
                      for f in os.listdir(label_dir)
                      if f.endswith(".ppm")]  # 判断是否是以ppm结尾
        for f in file_names:
            images.append(skimage.transform.resize(skimage.data.imread(f),(W,H)))  # data:提供一些测试图片和样本数据,imread读取图片
            labels.append(int(d))  # 标签与每个标签对应的图片数

    return images,labels
    #return np.asarray(images, np.float32), np.asarray(labels, np.int32)

images,labels=read_img(root_path)
print(labels)


images=np.asarray(images,np.float32)
# images=np.asarray(images)
labels=np.asarray(labels,np.float32)
for image in images[:4]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
# 打乱顺序
num_example = images.shape[0] #照片数
print(len(labels)) #类的数量
print("这是测试：",num_example)

arr = np.arange(num_example)
np.random.shuffle(arr)
data = images[arr]
label = labels[arr] #注意：从这里开始
print(label)


# 将所有数据分为训练集和验证集
ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]
#-----------------------------------------------------------------------
print(y_train)

# 构建神经网络前向传播
x = tf.placeholder(tf.float32, shape=[None, W,H,C], name='x')#输入数据集占位符，在执行Session时喂入
y= tf.placeholder(tf.int32, shape=[None, ], name='y')#输入标签集占位符，在执行Session时喂入
dropout=tf.placeholder(tf.float32,name='drop')##Dropout keep_drop占位符，在执行Session时喂入


def Forward(input,regularizer): #因为dropout正则方式只在训练时使用，所以提供bool型作为开关
    with tf.variable_scope('layer1-conv1'):#第一层，使用with tf.variable创建tf变量命名空间
        conv1_weights = tf.get_variable("weight", [5, 5, 3, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))#卷积核
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input, conv1_weights, strides=[1, 1, 1, 1], padding='SAME') #strides格式即为【1，s1，s2，，1】
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))   #32*32*32

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID") #16*16*32

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases)) #16*16*64

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') #8*64

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight", [3, 3, 64, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases)) #8*128

    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') #4*128

    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight", [3, 3, 128, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases)) #4*128


    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')#2*128
        nodes = 2* 2 * 128
        conv_end = tf.reshape(pool4, [-1, nodes]) #向量处理[none,512]
        '''函数的作用是将tensor变换为参数shape形式，其中的shape为一个列表形式，特殊的是列表可以实现逆序的遍历，
           即list(-1).-1所代表的含义是我们不用亲自去指定这一维的大小，函数会自动进行计算，但是列表中只能存在一个-1。
           （如果存在多个-1，就是一个存在多解的方程）'''


    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 2048],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1)) #[none,1024]
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights)) #tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表
        fc1_biases = tf.get_variable("bias", [2048], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(conv_end, fc1_weights) + fc1_biases)
        fc1=tf.nn.dropout(fc1,keep_prob=dropout)


    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [2048, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1)) #[none,512]
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        fc2=tf.nn.dropout(fc2,dropout)


    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 62], #[none,62],一共62类
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc3_biases = tf.get_variable("bias", [62], initializer=tf.constant_initializer(0.1))
        output = tf.matmul(fc2, fc3_weights) + fc3_biases #注意此处未执行激活函数，soft函数在损失（cost）函数处

    return output

#梯度下降、反向传播函数
#-------------------------------------------------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.01) #L2正则化  ,0.001为λ
output = Forward(x,regularizer) #先不涉及dropout
tf.add_to_collection('network-output', output)

y_onehot=tf.one_hot(y,depth=62,axis=1)
loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_onehot) #包含softmax的
#print(loss1)
loss2=tf.add_n(tf.get_collection('losses'))

loss=tf.add(loss2,loss1)
mean_loss=tf.reduce_mean(loss)

tf.summary.scalar('loss', tf.reduce_mean(loss)) #生成折线（scalar）图


global_step=tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate,global_step,30,decay_rate,staircase=True)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
correct_prediction = tf.equal(tf.cast(tf.argmax(output, 1), tf.int32), y) #tf.argmax 1为按行计算
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('acc', acc)
print(len(x_train)/batch_size)


#Mini_Batch
#--------------------------------------------------------------------
def minibatches(inputs=None, targets=None, batch_size=None):
    assert len(inputs) == len(targets) #判断是否和分类数相同
    '''python assert断言是声明其布尔值必须为真的判定，如果发生异常就说明表达示为假。
       可以理解assert断言语句为raise-if-not，用来测试表示式，其返回值为假，就会触发异常。'''
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size): #range(start, end, step
        excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
        '''yield减少内存压力，
        return 在返回结果后结束函数的运行，
        而yield 则是让函数变成一个生成器，生成器每次产生一个值（yield语句），函数被冻结，被唤醒后再产生一个值
'''


#会话
#---------------------------------------------------------------------
with tf.Session() as sess:
   # saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())#tensorflow变量初始化
    merged = tf.summary.merge_all()  # 绘图
    writer = tf.summary.FileWriter("logs/", sess.graph)

    writer_1 = tf.summary.FileWriter("./logs/plot_1")
    writer_2 = tf.summary.FileWriter("./logs/plot_2")

    for epoch in range(n_epoch):
        # training
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y: y_train_a,dropout:0.6})
            train_loss += err
            train_acc += ac
            n_batch += 1
            result = sess.run(merged, feed_dict={x: x_train_a, y: y_train_a, dropout: 0.6})
            writer_1.add_summary(result, epoch)
            writer_1.flush()

        print('这是第',epoch,'次训练')
        print("   train loss: %f" % (np.sum(train_loss) / n_batch))
        print("   train 准确率: %f" % (np.sum(train_acc) / n_batch))

        # validation
        val_loss=[]
        val_acc=[]
        n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size):
            err, ac = sess.run([mean_loss, acc], feed_dict={x: x_val_a, y: y_val_a,dropout:1})
            val_loss.append(err)
            val_acc.append(ac)
            #n_batch += 1
            result = sess.run(merged, feed_dict={x: x_val_a, y: y_val_a,dropout:1})
            writer_2.add_summary(result, epoch)
            writer_2.flush()

        print("   validation loss: %f" % (sess.run(tf.reduce_mean(val_loss))))
        print("   validation 准确率: %f" % (sess.run(tf.reduce_mean(val_acc))))
        print(sess.run([global_step,learning_rate]))
        #saver.save(sess, 'G:/traffic_data/Model/save_net')