from tkinter import *
from tkinter import filedialog
import  Test
def change():
    global  bm2,label2,file_path
    file_path = filedialog.askopenfilename()
    print(file_path)
    bm2 = PhotoImage(file=file_path)
    label2.configure(image = bm2)
    var.set(file_path)
    #print(Test.)
top = Tk()
top.geometry('500x500')
var=StringVar()
label1=Label(top,textvariable=var)
label1.pack()
bm = PhotoImage(file = 'G:/traffic_data/Training/00019/00015_00001.ppm')

label2 = Label(top, image = bm)
label2.pack()
button = Button(top, text = "changepicture", command = change)
button.pack()
#label3=Label(top,textvariable=,)
top.mainloop()