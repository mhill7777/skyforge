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
    
    #helps make plots
    def plot_VS_TargetValue(self, targetValue, tolerance_m=0.0005):
        graphData_error = [x-targetValue for x in self.dataList]
        numOfMeasurements=range(len(self.dataList))
        upper_bound = tolerance_m
        lower_bound = -1 * tolerance_m
        fig, ax = plt.subplots()
        plt.plot(graphData_error)
        # 2. Plot the target value line
        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Target Value')

        # 3. Plot the upper and lower tolerance lines (optional, can be done implicitly with fill_between)
        ax.axhline(upper_bound, color='gray', linestyle=':', linewidth=1, label='Tolerance Limit')
        ax.axhline(lower_bound, color='gray', linestyle=':', linewidth=1)

        # 4. Shade the tolerance band
        ax.fill_between(
            numOfMeasurements,
            lower_bound,
            upper_bound,
            color='red',
            alpha=0.2, # Adjust transparency
            label='Tolerance Range'
        )
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.show()