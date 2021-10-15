from PIL import Image
from tqdm import tqdm
import sys

import cv2

filePath = "/media/joao/My Passport/Elements/PointCloudsMaterialHash/LiDAR_PointCloud1/"

filePath_noBack = "/media/joao/My Passport/Elements/PointCloudsMaterialHash_withIntensity_v2/LiDAR_PointCloud1/"
input = cv2.imread(filePath_noBack + 'LiDAR_PointCloud_Camera_Print_Day_0_teste.jpg')

with open(filePath_noBack+"LiDAR_PointCloud.ply", 'r') as f:
    points_noBack = f.readlines()[8:]

with open(filePath_noBack+"LiDAR_PointCloud_points.txt", 'r') as f:
    points = f.readlines()

commom_points = []

# for x in points:
#     print(x[:-7])
#     break

# sys.exit(1)

for a, x in enumerate(points_noBack):
    points_noBack[a] = x[:-5]

for i in points_noBack:
    for x in points:
        if i in x:
            commom_points.append(x[-11:-3])
            break

height, width = input.shape[:2]

for i in commom_points:
    k = i.split(" ")
    if len(k) > 2:
        x = k[1]
        y = k[2]
    else:
        x = k[0]
        y = k[1]
    input[int(y), int(x)] = (0,0,255)

cv2.imshow('title',input)
cv2.waitKey(0)
cv2.destroyAllWindows()

sys.exit(1)

# Get input size
height, width = input.shape[:2]

print(height, width)

# Desired "pixelated" size
w, h = (16, 16)

# Resize input to "pixelated" size
temp = cv2.resize(input, (width, height), interpolation=cv2.INTER_LINEAR)

# Initialize output image
output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

cv2.imshow('Input', input)
cv2.imshow('Output', output)

cv2.waitKey(0)