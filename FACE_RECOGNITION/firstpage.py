from tkinter import *
import os
from datetime import datetime;
root=Tk()
root.configure(background="white")
def function1():

    os.system("py dataset_capture.py")
    
def function2():
    
    os.system("py training_dataSet.py")

def function3():

    os.system("py recognizer.py")
   
def function6():

    root.destroy()

def attend():
    os.startfile(os.getcwd()+"\\firebase\\attendance_files\\attendance"+str(datetime.now().date())+'.xls')

root.title("AUTOMATIC ATTENDANCE MANAGEMENT USING FACE RECOGNITION")

Label(root, text="FACE RECOGNITION ATTENDANCE SYSTEM",font=("times new roman",20),fg="white",bg="maroon",height=2).grid(row=0,rowspan=2,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

Button(root,text="Create Dataset",font=("times new roman",20),bg="#0D47A1",fg='white',command=function1).grid(row=3,columnspan=2,sticky=W+E+N+S,padx=5,pady=5)

Button(root,text="Train Dataset",font=("times new roman",20),bg="#0D47A1",fg='white',command=function2).grid(row=4,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

Button(root,text="Recognize + Attendance",font=('times new roman',20),bg="#0D47A1",fg="white",command=function3).grid(row=5,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

Button(root,text="Attendance Sheet",font=('times new roman',20),bg="#0D47A1",fg="white",command=attend).grid(row=6,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

Button(root,text="Exit",font=('times new roman',20),bg="maroon",fg="white",command=function6).grid(row=9,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

root.mainloop()
