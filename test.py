import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


a = np.array([[1,1,-1],[4,0,1],[0,4,0]])
print(a)
a = np.linalg.inv(a)
print(a)