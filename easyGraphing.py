import numpy as np
import matplotlib.pyplot as plt
import math

class dataGrapher:
    def __init__(self, title, x_label, y_label):
        self.dataList=[]
        self.title=title
        self.x_label=x_label
        self.y_label = y_label

    def append(self,data):
        self.dataList.append(data)

    def graph(self):
        plt.plot(self.dataList)
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        # plt.show()