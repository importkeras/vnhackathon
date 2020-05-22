from __future__ import division, absolute_import
from PIL import ImageTk, Image
from timeit import time
from tkinter import *
from tkinter.ttk import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import ImageTk, Image
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from glob import glob
from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import Canvas
from tkinter.scrolledtext import ScrolledText
from tkcalendar import Calendar, DateEntry
from timeit import time
from timeit import default_timer as timer  
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import os
import sys
import datetime
import random
import torch
import cv2
import time
import numpy as np
import tkinter as tk
import colorsys
import csv

def _get_class():
        global classes_path
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

def _get_anchors():
        global anchors_path
        anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

def generate():
        global input_image_shape
        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        random.seed(10101)  
        random.shuffle(colors)  
        random.seed(None) 
        input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(yolo_model.output, anchors,
                len(class_names), input_image_shape,
                score_threshold=score, iou_threshold=iou)
        return boxes, scores, classes

def detect_image(image):
        global input_image_shape
        if is_fixed_size:
            boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype = "float32")
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  
        
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        return_boxs = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            if predicted_class != "person":
                continue
            box = out_boxes[i]
            x = int(box[1])  
            y = int(box[0])  
            w = int(box[3]-box[1])
            h = int(box[2]-box[0])
            if x < 0 :
                w = w + x
                x = 0
            if y < 0 :
                h = h + y
                y = 0 
            return_boxs.append([x,y,w,h])

        return return_boxs

def close_session():
        sess.close()

yellowCount = np.array([])
redCount = np.array([])
groupMax= np.array([])
model_path = "model_data/yolo.h5"
anchors_path = "model_data/yolo_anchors.txt"
classes_path = "model_data/coco_classes.txt"
model_path = os.path.expanduser(model_path)
yolo_model = load_model(model_path, compile=False)
score = 0.5
iou = 0.5
class_names = _get_class()
anchors = _get_anchors()
sess = K.get_session()
model_image_size = (416, 416)
input_image_shape = [0, 0]
is_fixed_size = model_image_size != (None, None)
boxes, scores, classes = generate()
model_filename = "model_data/mars-small128.pb"
encoder = gdet.create_box_encoder(model_filename, batch_size = 1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.3, None)
tracker = Tracker(metric)
vectorGroup1 = []
vectorGroup2 = []
vectorCountYellow = []
vectorCountRed = []

def move_window(event):
    root.geometry("+{0}+{1}".format(event.x_root, event.y_root))
    
def change_on_hovering(event):
    global close_button
    close_button["bg"] = "red"
    
def return_to_normalstate(event):
    global close_button
    close_button["bg"] = "#2e2e2e"

def dist(x1, y1, x2, y2):
        return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)

def pairFigure():
        global yellowCount, redCount
        with open("output/group.csv", "r") as file:
                reader = csv.reader(file)
                count = 0
                for row in reader:
                        count += 1
                        if (count % 2 == 1):
                                yellowCount = np.append(yellowCount, int(row[2]))
                                redCount = np.append(redCount, int(row[3]))
        dateAxis = np.array([])
        valueAxis = np.row_stack((yellowCount, redCount))
        maxAxis = max(np.amax(yellowCount), np.amax(redCount))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(valueAxis[0, :], label = "Số lượng cặp người có khoảng cách gần nhau", color = "y")
        ax1.plot(valueAxis[1, :], label = "Số lượng cặp người có khoảng cách rất gần nhau", color = "r")
        axes = plt.axes()
        maxAxis = max(5, max(np.amax(yellowCount), np.amax(redCount)))
        axes.set_ylim([0, maxAxis])
        axesNumber = []
        for i in range(int(maxAxis) + 1): axesNumber.append(i)
        axes.set_yticks(axesNumber)
        plt.xticks(dateAxis)
        plt.xlabel("Biểu đồ số lượng cặp người có khoảng cách gần và rất gần nhau theo thời gian")
        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(loc = "best")
        ax1.grid('on')
        plt.savefig("output/pairFigure.jpg")

def groupFigure():
        global groupMax
        with open("output/group.csv", "r") as file:
                reader = csv.reader(file)
                count = 0
                for row in reader:
                        count += 1
                        if (count % 2 == 1):
                                groupMax = np.append(groupMax, int(row[0]))
        dateAxis = np.array([])
        maxAxis = np.amax(groupMax)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(groupMax, label = "Số lượng người trong nhóm lớn nhất", color = "b")
        axes = plt.axes()
        maxAxis = np.amax(groupMax)
        axes.set_ylim([0, maxAxis])
        axesNumber = []
        for i in range(int(maxAxis) + 1): axesNumber.append(i)
        axes.set_yticks(axesNumber)
        plt.xticks(dateAxis)
        plt.xlabel("Biểu đồ số lượng người trong nhóm lớn nhất theo thời gian")
        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(loc = "best")
        ax1.grid('on')
        plt.savefig("output/groupFigure.jpg")

def openVideo():
        global countFrame, vectorGroup1, vectorGroup2, vectorCountYellow, vectorCountRed
        filename = filedialog.askopenfilename(initialdir = "/home/", title = "Chọn video", filetypes = [("Video", [".mp4", ".avi"] )])
        directoryText.set(" " + filename)
        videoCapture = cv2.VideoCapture(filename)
        w = int(videoCapture.get(3))
        h = int(videoCapture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter("result.avi", fourcc, 15, (w, h))
        outGroup1 = cv2.VideoWriter("outGroup1.avi", cv2.VideoWriter_fourcc(*'DIVX'), 15, (180, 180))
        outGroup2 = cv2.VideoWriter("outGroup2.avi", cv2.VideoWriter_fourcc(*'DIVX'), 15, (180, 180))
        countFrame = 0
        statusText.set(" Đang phân tích")
        def videoStream():
                global countFrame, vectorGroup1, vectorGroup2, vectorCountYellow, vectorCountRed
                ret, frame = videoCapture.read()
                if (ret != True):
                        return
                countFrame += 1
                image = Image.fromarray(frame[..., ::-1])
                boxs = detect_image(image)
                features = encoder(frame, boxs)
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
                detections = [detections[i] for i in indices]
                tracker.predict()
                tracker.update(detections)
                det = [[] for _ in range(len(detections))]
                for i in range(len(detections)):
                        det[i] = detections[i]
                        bbox = det[i].to_tlbr()
                        cv2.circle(frame, (int((bbox[0] + bbox[2]) // 2), int((bbox[1] + bbox[3]) // 2)), 3, (0, 255, 0), 3)
                        l = int(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
                edge = [[] for _ in range(len(detections))]
                countRed = 0
                countYellow = 0
                for i in range(len(detections)):
                        for j in range(i + 1, len(detections)):
                                bbox1 = detections[i].to_tlbr()
                                bbox2 = detections[j].to_tlbr()
                                x1 = int((bbox1[0] + bbox1[2]) // 2)
                                y1 = int((bbox1[1] + bbox1[3]) // 2)
                                x2 = int((bbox2[0] + bbox2[2]) // 2)
                                y2 = int((bbox2[1] + bbox2[3]) // 2)
                                if (min(y1, y2) <= h // 3):
                                        if (dist(x1, y1, x2, y2) <= 30 * 30):
                                                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                                edge[i].append(j)
                                                edge[j].append(i)
                                                countRed += 1
                                        else:
                                                if (dist(x1, y1, x2, y2) <= 100 * 100):
                                                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                                                        edge[i].append(j)
                                                        edge[j].append(i)
                                                        countYellow += 1
                                else:
                                        if (min(y1, y2) <= h // 2):
                                                if (dist(x1, y1, x2, y2) <= 100 * 100):
                                                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                                        edge[i].append(j)
                                                        edge[j].append(i)
                                                        countRed += 1
                                                else:
                                                        if (dist(x1, y1, x2, y2) <= 150 * 150):
                                                                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                                                                edge[i].append(j)
                                                                edge[j].append(i)
                                                                countYellow += 1
                                        else:
                                                if (dist(x1, y1, x2, y2) <= 130 * 130):
                                                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                                        edge[i].append(j)
                                                        edge[j].append(i)
                                                        countRed += 1
                                                else:
                                                        if (dist(x1, y1, x2, y2) <= 250 * 250):
                                                                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                                                                edge[i].append(j)
                                                                edge[j].append(i)
                                                                countYellow += 1
                cv2.putText(frame, "So luong cap gan nhau: " + str(countYellow), (int(w - 650), int(h - 70)), 0, 5e-3 * 200, (0, 255, 255), 3)
                cv2.putText(frame, "So luong cap rat gan nhau: " + str(countRed), (int(w - 650), int(h - 40)), 0, 5e-3 * 200, (0, 0, 255), 3)
                vectorCountYellow.append(countYellow)
                vectorCountRed.append(countRed)
                out.write(frame)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                cv2image = cv2.resize(cv2image, (755, 520))
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image = img)
                lmain.imgtk = imgtk
                lmain.configure(image = imgtk)
                mark = [True for _ in range(len(detections))]
                component = [[[], 0] for _ in range(len(detections))]
                componentCount = -1
                def DFS(u):
                        component[componentCount][0].append(u)
                        mark[u] = False
                        for v in edge[u]:
                                if (mark[v] == True): DFS(v)
                for i in range(len(detections)):
                        if (mark[i] == True):
                                componentCount += 1
                                DFS(i)
                for i in range(componentCount):
                        component[i][1] = len(component[i][0])
                def takeSecond(elem):
                        return elem[1]
                component.sort(key = takeSecond, reverse = True)
                if (componentCount >= 1):
                        xMin = 100000
                        yMin = 100000
                        xMax = 0
                        yMax = 0
                        for i in component[0][0]:
                                bbox = det[i].to_tlbr()
                                xMax = max(xMax, max(bbox[0], bbox[2]))
                                xMin = min(xMin, min(bbox[0], bbox[2]))
                                yMax = max(yMax, max(bbox[1], bbox[3]))
                                yMin = min(yMin, min(bbox[1], bbox[3]))
                        cropFrame = frame[int(yMin) : int(yMax), int(xMin) : int(xMax)]
                        cv2.imwrite("temp.jpg", cropFrame)
                        cv2image13 = cv2.imread("temp.jpg")
                        cv2image13 = cv2.resize(cv2image13, (180, 180))
                        outGroup1.write(cv2image13)
                        cv2image13 = cv2.cvtColor(cv2image13, cv2.COLOR_BGR2RGBA)
                        img13 = Image.fromarray(cv2image13)
                        imgtk13 = ImageTk.PhotoImage(image = img13)
                        lmain13.imgtk13 = imgtk13
                        lmain13.configure(image = imgtk13)
                        group1Text.set(" Số người trong nhóm: " + str(len(component[0][0])))
                        vectorGroup1.append(len(component[0][0]))
                else:
                        cv2image13 = cv2.imread("avatar.jpg")
                        cv2image13 = cv2.resize(cv2image13, (180, 180))
                        outGroup1.write(cv2image13)
                        vectorGroup1.append(0)
                if (componentCount >= 2):
                        xMin = 100000
                        yMin = 100000
                        xMax = 0
                        yMax = 0
                        for i in component[1][0]:
                                bbox = det[i].to_tlbr()
                                xMax = max(xMax, max(bbox[0], bbox[2]))
                                xMin = min(xMin, min(bbox[0], bbox[2]))
                                yMax = max(yMax, max(bbox[1], bbox[3]))
                                yMin = min(yMin, min(bbox[1], bbox[3]))
                        cropFrame = frame[int(yMin) : int(yMax), int(xMin) : int(xMax)]
                        cv2.imwrite("temp.jpg", cropFrame)
                        cv2image14 = cv2.imread("temp.jpg")
                        cv2image14 = cv2.resize(cv2image14, (180, 180))
                        outGroup2.write(cv2image14)
                        cv2image14 = cv2.cvtColor(cv2image14, cv2.COLOR_BGR2RGBA)
                        img14 = Image.fromarray(cv2image14)
                        imgtk14 = ImageTk.PhotoImage(image = img14)
                        lmain14.imgtk14 = imgtk14
                        lmain14.configure(image = imgtk14)
                        group2Text.set(" Số người trong nhóm: " + str(len(component[1][0])))
                        vectorGroup2.append(len(component[1][0]))
                else:
                        cv2image14 = cv2.imread("avatar.jpg")
                        cv2image14 = cv2.resize(cv2image14, (180, 180))
                        outGroup2.write(cv2image14)
                        vectorGroup2.append(0)
                progress["value"] += 100 / int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
                if (int(countFrame) == int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))):
                        out.release()
                        outGroup1.release()
                        outGroup2.release()
                        statusText.set(" Đã phân tích xong")
                        progress["value"] = 100
                        rows = zip(vectorGroup1, vectorGroup2, vectorCountYellow, vectorCountRed)
                        with open("group.csv", "w") as f:
                                writer = csv.writer(f)
                                for row in rows:
                                        writer.writerow(row)
                        pairFigure()
                        groupFigure()
                        return
                else:
                        lmain.after(1, videoStream)
        videoStream()

def playVideo():
        global vector, pos
        filename = filedialog.askopenfilename(initialdir = "/home/", title = "Chọn video", filetypes = [("Video", [".mp4", ".avi"] )])
        directoryText.set(" " + filename)
        statusText.set(" Đang phân tích")
        pos = 0
        with open("output/group.csv", "r") as f:
                reader = csv.reader(f)
                pos = 0
                for row in reader:
                        pos += 1
                        if (pos % 2 == 1):
                                vector.append(row)
        videoCapture = cv2.VideoCapture("output/result.avi")
        pos = -1
        def videoStream():
                global pos, vector
                ret, frame = videoCapture.read()
                pos += 1
                if (ret != True):
                        statusText.set(" Đã phân tích xong")
                        progress["value"] = 100
                        return
                group1Text.set(" Số người trong nhóm: " + str(vector[pos][0]))
                group2Text.set(" Số người trong nhóm: " + str(vector[pos][1]))
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                cv2image = cv2.resize(cv2image, (755, 520))
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image = img)
                lmain.imgtk = imgtk
                lmain.configure(image = imgtk)
                progress["value"] += 100 / int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
                lmain.after(1, videoStream)
        videoStream()

def choosingName():
    imageList5.delete(0, END)
    imageList5.insert(END, "18/5/2020")
    imageList5.insert(END, "19/5/2020")
    imageList5.insert(END, "20/5/2020")
    imageList5.insert(END, "21/5/2020")
            
root = Tk()
root.overrideredirect(True)
root.title("Hệ thống hỗ trợ giám sát chấp hành giãn cách xã hội phòng chống dịch bệnh truyền nhiễm cho doanh nghiệp") 
root.geometry('1230x725')
root.resizable(0, 0)
title_bar = tk.Frame(root, bg = "#2e2e2e", relief = "raised", bd = 2, height = 30, highlightthickness = 0)
title_bar.pack(side = TOP, fill = "both")
close_button = tk.Button(title_bar, text = 'X', command = root.destroy, bg = "#2e2e2e", padx = 2, pady = 2, activebackground = "red", bd = 0, font = "bold", fg = "white", highlightthickness = 0)
close_button.pack(side = RIGHT)
title = Label(title_bar, text = " ")
title.config(font = ("Courier", 12), background = "#2e2e2e", foreground = "#ffffff")
title.pack(side = LEFT)
app3 = Frame(title_bar)
app3.pack(side = LEFT)
lmain3 = Label(app3)
lmain3.grid(padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + S)
cv2image3 = cv2.imread("icon.png")
cv2image3 = cv2.cvtColor(cv2image3, cv2.COLOR_BGR2RGBA)
cv2image3 = cv2.resize(cv2image3, (15, 15))
img3 = Image.fromarray(cv2image3)
imgtk3 = ImageTk.PhotoImage(image = img3)
lmain3.imgtk3 = imgtk3
lmain3.configure(image = imgtk3)
title = Label(title_bar, text = " Import Keras - Vietnam Online Hackathon")
title.config(font = ("Courier", 12), background = "#2e2e2e", foreground = "#ffffff")
title.pack(side = LEFT)
title_bar.bind("<B1-Motion>", move_window)
close_button.bind("<Enter>", change_on_hovering)
close_button.bind("<Leave>", return_to_normalstate)
app7 = Frame(root)
app7.pack(side = TOP, fill = "both")
lmain7 = Label(app7)
lmain7.grid(padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + S)
cv2image7 = cv2.imread("banner.png")
cv2image7 = cv2.cvtColor(cv2image7, cv2.COLOR_BGR2RGBA)
cv2image7 = cv2.resize(cv2image7, (1226, 100))
img7 = Image.fromarray(cv2image7)
imgtk7 = ImageTk.PhotoImage(image = img7)
lmain7.imgtk7 = imgtk7
lmain7.configure(image = imgtk7)
tabParent = ttk.Notebook(root)
class1 = ttk.Frame(tabParent)
class2 = ttk.Frame(tabParent)
class3 = ttk.Frame(tabParent)
class4 = ttk.Frame(tabParent)
tabParent.add(class1, text = "Quan sát")
tabParent.add(class2, text = "Thống kê")
tabParent.add(class3, text = "Quản lý nhân viên")
tabParent.add(class4, text = "Quản lý khách hàng")
tabParent.pack(side = TOP, expand = 1, fill = "both")
videoFrame = LabelFrame(class1, text = "Hình ảnh trực tiếp từ camera quan sát", width = 775, height = 540)
videoFrame.grid(row = 1, rowspan = 4, column = 1, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = W + N + S)
settingFrame = LabelFrame(class1, text = "Hệ thống", width = 210)
settingFrame.grid(row = 1, column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 5, sticky = W + E + N)
infoFrame = LabelFrame(class1, text = "Thông tin", width = 210)
infoFrame.grid(row = 2, rowspan = 2, column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = W + E + N)
exportFrame = LabelFrame(class1, text = "Xuất", width = 210)
exportFrame.grid(row = 4, column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = W + E + N)
groupFrame = LabelFrame(class1, text = "Nhóm nhiều người gần nhau nhất", width = 195)
groupFrame.grid(row = 1, rowspan = 2, column = 2, padx = 5, pady = 5, ipadx = 0, ipady = 5, sticky = E + W + N + S)
Button(exportFrame, text = "Dữ liệu", width = 14).grid(row = 1, column = 1, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
Button(exportFrame, text = "Video", width = 14).grid(row = 1, column = 2, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
app9 = Frame(class1)
app9.grid(row = 3, rowspan = 2, column = 2, padx = 5, pady = 0, ipadx = 0, ipady = 0, sticky = E + W + N + S)
lmain9 = Label(app9)
lmain9.grid(padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + S)
cv2image9 = cv2.imread("logo.jpg")
cv2image9 = cv2.cvtColor(cv2image9, cv2.COLOR_BGR2RGBA)
cv2image9 = cv2.resize(cv2image9, (205, 70))
img9 = Image.fromarray(cv2image9)
imgtk9 = ImageTk.PhotoImage(image = img9)
lmain9.imgtk9 = imgtk9
lmain9.configure(image = imgtk9)
lmain = Label(videoFrame)
lmain.grid(padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
cv2image = cv2.imread("default.png")
cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
cv2image = cv2.resize(cv2image, (755, 520))
img = Image.fromarray(cv2image)
imgtk = ImageTk.PhotoImage(image = img)
lmain.imgtk = imgtk
lmain.configure(image = imgtk)
Button(settingFrame, text = "Mở tập tin video", command = playVideo).grid(row = 1, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = W + E)
Label(settingFrame, text = " Đường dẫn tập tin").grid(row = 3, sticky = W)
directoryText = StringVar()
directory = Entry(settingFrame, width = 32, textvariable = directoryText)
directory.grid(row = 4, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = W)
Label(settingFrame, text = " Trạng thái phân tích").grid(row = 6, sticky = W)
statusText = StringVar()
status = Entry(settingFrame, width = 32, textvariable = statusText)
status.grid(row = 7, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = W)
statusText.set(" Đang chờ tập tin")
Label(settingFrame, text = " Ngày giờ hệ thống").grid(row = 9, sticky = W)
curTime = Entry(settingFrame, width = 32)
curTime.insert(15, time.strftime(" %m/%d/%Y, %H:%M:%S %p"))
curTime.grid(row = 10, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = W)
progress = Progressbar(settingFrame, orient = HORIZONTAL, mode = "determinate")
progress.grid(row = 11, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = N + W + E + S)
progress["value"] = 0
app8 = Frame(infoFrame)
app8.pack(side = TOP, fill = "both")
lmain8 = Label(app8)
lmain8.grid(padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + S)
cv2image8 = cv2.imread("info.png")
cv2image8 = cv2.cvtColor(cv2image8, cv2.COLOR_BGR2RGBA)
cv2image8 = cv2.resize(cv2image8, (195, 200))
img8 = Image.fromarray(cv2image8)
imgtk8 = ImageTk.PhotoImage(image = img8)
lmain8.imgtk8 = imgtk8
lmain8.configure(image = imgtk8)
countFrame = 0
pos = 0
vector = []
app13 = Frame(groupFrame)
app13.grid(row = 1, padx = 5, pady = 2, ipadx = 0, ipady = 0, sticky = W + E + N + S)
lmain13 = Label(app13)
lmain13.grid(row = 1, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + S)
cv2image13 = cv2.imread("avatar.jpg")
cv2image13 = cv2.cvtColor(cv2image13, cv2.COLOR_BGR2RGBA)
cv2image13 = cv2.resize(cv2image13, (180, 180))
img13 = Image.fromarray(cv2image13)
imgtk13 = ImageTk.PhotoImage(image = img13)
lmain13.imgtk13 = imgtk13
lmain13.configure(image = imgtk13)
group1Text = StringVar()
group1 = Entry(groupFrame, textvariable = group1Text)
group1.grid(row = 2, padx = 10, pady = 2, ipadx = 0, ipady = 0, sticky = W + E + N + S)
group1Text.set(" Số người trong nhóm: ")
app14 = Frame(groupFrame)
app14.grid(row = 3, padx = 5, pady = 2, ipadx = 0, ipady = 0, sticky = W + E + N + S)
lmain14 = Label(app14)
lmain14.grid(row = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + S)
cv2image14 = cv2.imread("avatar.jpg")
cv2image14 = cv2.cvtColor(cv2image14, cv2.COLOR_BGR2RGBA)
cv2image14 = cv2.resize(cv2image14, (180, 180))
img14 = Image.fromarray(cv2image14)
imgtk14 = ImageTk.PhotoImage(image = img14)
lmain14.imgtk14 = imgtk14
lmain14.configure(image = imgtk14)
group2Text = StringVar()
group2 = Entry(groupFrame, textvariable = group2Text)
group2.grid(row = 4, padx = 10, pady = 2, ipadx = 0, ipady = 0, sticky = W + E + N + S)
group2Text.set(" Số người trong nhóm: ")
app16 = Frame(class2)
app16.grid(row = 1, column = 1, padx = 60, pady = 25, ipadx = 0, ipady = 0, sticky = W + E + N + S)
lmain16 = Label(app16)
lmain16.grid(row = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + S)
cv2image16 = cv2.imread("output/pairFigure.jpg")
cv2image16 = cv2.cvtColor(cv2image16, cv2.COLOR_BGR2RGBA)
cv2image16 = cv2.resize(cv2image16, (500, 500))
img16 = Image.fromarray(cv2image16)
imgtk16 = ImageTk.PhotoImage(image = img16)
lmain16.imgtk16 = imgtk16
lmain16.configure(image = imgtk16)
app17 = Frame(class2)
app17.grid(row = 1, column = 2,  padx = 5, pady = 25, ipadx = 0, ipady = 0, sticky = W + E + N + S)
lmain17 = Label(app17)
lmain17.grid(row = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + S)
cv2image17 = cv2.imread("output/groupFigure.jpg")
cv2image17 = cv2.cvtColor(cv2image17, cv2.COLOR_BGR2RGBA)
cv2image17 = cv2.resize(cv2image17, (500, 500))
img17 = Image.fromarray(cv2image17)
imgtk17 = ImageTk.PhotoImage(image = img17)
lmain17.imgtk17 = imgtk17
lmain17.configure(image = imgtk17)
filterDateFrame = LabelFrame(class4, text = "Lịch sử")
filterDateFrame.grid(row = 2, column = 2, padx = 350, pady = 25, ipadx = 0, ipady = 0, sticky = E + W + N + S)
app5 = Frame(filterDateFrame)
app5.grid(row = 1, column = 1, columnspan = 2, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
lmain5 = Label(app5)
lmain5.grid(padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + W + N + S)
cv2image5 = cv2.imread("null.png")
cv2image5 = cv2.cvtColor(cv2image5, cv2.COLOR_BGR2RGBA)
cv2image5 = cv2.resize(cv2image5, (450, 450))
img5 = Image.fromarray(cv2image5)
imgtk5 = ImageTk.PhotoImage(image = img5)
lmain5.imgtk = imgtk5
lmain5.configure(image = imgtk5)
imageList5 = Listbox(filterDateFrame, width = 5, height = 15, font = ('times', 13))
imageList5 = tk.Listbox(filterDateFrame, width = 8, height = 15, font = ('times', 13))
imageList5.grid(row = 1, column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
imageList5.bind("<ButtonRelease-1>")
mynumber = tk.StringVar()
combobox = ttk.Combobox(filterDateFrame, textvariable = mynumber)
combobox["values"] = tuple(["Pham Huy", "Thiet Gia", "Minh Nhat", "Anh Hao"])
combobox.grid(row = 2, column = 1, columnspan = 2, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
button = ttk.Button(filterDateFrame, text = "Chọn tên", command = choosingName)
button.grid(row = 2, column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
