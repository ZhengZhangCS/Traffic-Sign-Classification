from skimage import io, transform
import tensorflow as tf
import numpy as np
from tkinter import *
from tkinter import filedialog
#模型读出后 outputs可追加softmax


top = Tk()
top.title('深度学习交通标志图像分类器')
top.geometry('600x500')

canvas=Canvas(top,bg='blue',height=5000,width=500)
image_file=PhotoImage(file='timg.gif')
canvas.create_image(0,0,anchor='nw',image=image_file)
canvas.pack()



W = 32
H= 32
C= 3



top.mainloop()