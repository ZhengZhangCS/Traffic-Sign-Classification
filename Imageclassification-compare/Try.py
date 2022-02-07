from skimage import io, transform
import tensorflow as tf
import numpy as np
from tkinter import *
from tkinter import filedialog
#模型读出后 outputs可追加softmax
traffic_dict={0:'路面不平',1:'前方坡道',2:'易滑',3:'向左急弯'
,4:'向右急弯'
,5:'反向弯路1'
,6:'反向弯路2'
,7:'注意行人'
,8:'注意非机动车'
,9:'注意牲畜'
,10:'注意前方施工'
,11:'注意前方交通灯'
,12:'有人看守铁道路口'
,13:'注意危险'
,14:'注意两侧变窄'
,15:'注意左侧变窄'
,16:'注意右侧变窄'
,17:'干路先行'
,18:'交叉路段'
,19:'重点注意'
,20:'会车让行'
,21:'STOP'
,22:'禁止驶入'
,23:'禁止非机动车'
,24:'限重3.5T'
,25:'禁止卡车通行'
,26:'限宽警告'
,27:'限高警告'
,28:'禁止通行'
,29:'禁止左拐'
,30:'禁止右拐'
,31:'注意超车'
,32:'限速警告'
,33:'允许行人与非机动车'
,34:'单行道'
,35:'单行路向左'
,36:'直行和有转合用车道 '
,37:'环岛行驶'
,38:'非机动车道'
,39:'行人与非机动车共道'
,40:'禁止车辆长时停放'
,41:'禁止车辆临时或长时停放'
,42:'1/15'
,43:'16/31'
,44:'会车先行'
,45:'允许停泊'
,46:'允许特殊人士停泊'
,47:'允许机动车停泊'
,48:'允许载物货车停泊'
,49:'允许客车停泊'
,50:'占用停泊位'
,51:'允许道路活动'
,52:'禁止道路活动'
,53:'直行'
,54:'道路截止'
,55:'禁止施工道路'
,56:'前方人行道'
,57:'前方非机动车道',58:'右转停泊',59:'高突路面',60:'解除主道优先',61:'主道优先'
}
def read_one_image(path):
    img = io.imread(path) #图片绝对地址
    img = transform.resize(img, (W,H,C))
    return np.asarray(img)

def change():
    global  bm2,label2,file_path,data,data1,saver,pred,graph,x,dropout,feed_dict,classification_result,result,var2,traffic_dict
    file_path = filedialog.askopenfilename()
    print(file_path)
    bm2 = PhotoImage(file=file_path)
    label2.configure(image = bm2)
    var.set(file_path)

    def read_one_image(path):
        img = io.imread(path)  # 图片绝对地址
        img = transform.resize(img, (W, H, C))
        return np.asarray(img)

    with tf.Session() as sess:
        data = []
        data1 = read_one_image(file_path)
        data.append(data1)

        saver = tf.train.import_meta_graph('G:/traffic_data/Model/save_net.meta')  # 导入模型
        saver.restore(sess, tf.train.latest_checkpoint('G:/traffic_data/Model/'))
        pred = tf.get_collection('network-output')[0]

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        dropout = graph.get_tensor_by_name('drop:0')
        feed_dict = {x: data, dropout: 1}

        classification_result = sess.run(pred, feed_dict)

        # 打印出预测矩阵
        print(classification_result)
        # 打印出预测矩阵每一行最大值的索引
        print(tf.argmax(classification_result, 1).eval())
        result=tf.argmax(classification_result, 1).eval()
        for i in result:
            print(traffic_dict[i])
            var2.set(traffic_dict[i])

top = Tk()
top.title('深度学习交通标志图像分类器')
top.geometry('500x500')
canvas=Canvas(top,bg='blue',height=500,width=500)
image_file=PhotoImage(file='title.ppm')
canvas.create_image(0,0,anchor='nw',image=image_file)
canvas.pack()

var=StringVar()
var2=StringVar()
label1=Label(top,textvariable=var)
label1.pack()
bm = PhotoImage(file = 'G:/traffic_data/Training/00019/00015_00001.ppm')

label2 = Label(top, image = bm,width=128,height=128)
label2.pack()
button = Button(top, text = "请选择图片分类", command = change)
button.pack()
label3=Label(top,textvariable=var2)
label3.pack()

label4=Label(top,text='姓名：张政\n'
                      '专业：物联网工程\n'
                      '毕业设计名称：图像分类算法设计与实现研究',justify=LEFT) #标签换行，左对齐
label4.place(x=300,y=400,anchor='nw')
canvas.create_window(250, 100,anchor='center', window=label1)
canvas.create_window(250, 200, window=label2)
canvas.create_window(250, 300, window=button)
canvas.create_window(250, 400, window=label3)
canvas.create_window(450, 450, window=label4)


W = 32
H= 32
C= 3



top.mainloop()