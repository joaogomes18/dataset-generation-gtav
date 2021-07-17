# 
# This python script colorizes the uncolored point clouds (LiDAR_PointCloud.ply and LiDAR_PointCloud_error.ply).
# However these uncolored point clouds are not used by this script in order the produce the colored point clouds.
# Instead, it used the files LiDAR_PointCloud_points.txt and LiDAR_PointCloud_error.txt to do so.
# For each directory (named LiDAR_PointCloudX) holding point cloud data, this script is called 2 times:
#	- first, to produce the ideal colored point cloud, by specifying the file LiDAR_PointCloud_points.txt in the command argument
#	- second, to produce the colored point cloud with errors/noise, by specifying the file LiDAR_PointCloud_error.txt in the command argument
# This script produces 3 colored point clouds: one for the day, one for the night and one for the cloudy weather.
from PIL import Image
from tqdm import tqdm
import sys

# arg = sys.argv[1]

# splits the argument by '\'. The argument can be a file name, a relative or full path to the file
filePath = "/home/joao/Desktop/LiDAR_PointCloud2/"

# obtain the directory path to where the file specified in the argument resides.
# discards the last elment because it's the name of the text file
# dirPath = ''
# for i in range(0, len(filePath)-1):
# 	dirPath += filePath[i] + '\\'

# load the pictures that into memory in order to assign them to them points of the point cloud
# imDay = []
# imDay.append(Image.open(dirPath + 'LiDAR_PointCloud_Camera_Print_Day_0.bmp', 'r'))
# imDay.append(Image.open(filePath + 'LiDAR_PointCloud_Camera_Print_Day_0_test.jpg'))
imDay = Image.open(filePath + 'LiDAR_PointCloud_Camera_Print_Day_0_test.jpg')
# imDay.append(Image.open(dirPath + 'LiDAR_PointCloud_Camera_Print_Day_1.bmp', 'r'))
# imDay.append(Image.open(dirPath + 'LiDAR_PointCloud_Camera_Print_Day_2.bmp', 'r'))

# imNight = []
# imNight.append(Image.open(dirPath + 'LiDAR_PointCloud_Camera_Print_Night_0.bmp', 'r'))
# imNight.append(Image.open(dirPath + 'LiDAR_PointCloud_Camera_Print_Night_1.bmp', 'r'))
# imNight.append(Image.open(dirPath + 'LiDAR_PointCloud_Camera_Print_Night_2.bmp', 'r'))

# imCloudy = []
# imCloudy.append(Image.open(dirPath + 'LiDAR_PointCloud_Camera_Print_Cloudy_0.bmp', 'r'))
# imCloudy.append(Image.open(dirPath + 'LiDAR_PointCloud_Camera_Print_Cloudy_1.bmp', 'r'))
# imCloudy.append(Image.open(dirPath + 'LiDAR_PointCloud_Camera_Print_Cloudy_2.bmp', 'r'))

# pix_rgb_day = []
# pix_rgb_day.append(imDay[0].load())
# pix_rgb_day.append(imDay[1].load())
# pix_rgb_day.append(imDay[2].load())

# pix_rgb_night = []
# pix_rgb_night.append(imNight[0].load())
# pix_rgb_night.append(imNight[1].load())
# pix_rgb_night.append(imNight[2].load())

# pix_rgb_cloudy = []
# pix_rgb_cloudy.append(imCloudy[0].load())
# pix_rgb_cloudy.append(imCloudy[1].load())
# pix_rgb_cloudy.append(imCloudy[2].load())

# open the file specified in the python argument
# file = open(sys.argv[1], 'r')
file = open(filePath + "LiDAR_PointCloud_points.txt", 'r')
# and count the number of lines that it has
n_lines = 0
for x in file:
    n_lines+=1
file.close()

# grayScale = []

# file = open(filePath + "LiDAR_PointCloud_points.txt", 'r')
# for line in tqdm(file, total=n_lines):
#     line = line.split()
#     if(line[3] != '0' and line[4] != '0'):
#         rgbDay = imDay.getpixel((int(line[3]),int(line[4])))
#         newRGBDay = [x/3 for x in rgbDay]
#         # grayScale.append(newRGBDay[0] + newRGBDay[1] + newRGBDay[2])
#         grayScale = (newRGBDay[0] + newRGBDay[1] + newRGBDay[2])/3
#         imDay.putpixel((int(line[3]),int(line[4])), (int(grayScale), int(grayScale), int(grayScale)))
#         # pix_rgb_day[0][int(line[3]),int(line[4])] = (int(grayScale), int(grayScale), int(grayScale))

# print(grayScale)

width, height = imDay.size

# Process every pixel
for x in range(width):
    for y in range(height):
        rgbDay = imDay.getpixel((x,y))
        newRGBDay = [x/3 for x in rgbDay]
        # grayScale.append(newRGBDay[0] + newRGBDay[1] + newRGBDay[2])
        grayScale = (newRGBDay[0] + newRGBDay[1] + newRGBDay[2])/3
        imDay.putpixel((x,y), (int(grayScale), int(grayScale), int(grayScale)))

imDay.close()


# file.close()

# closing and opening the file again to reset the pointer
# file = open(sys.argv[1], 'r')
# color_mapping = {}

# print("--" + filePath[len(filePath)-1][:-4])
# with open(dirPath + filePath[len(filePath)-1][:-4] + "_PC_Day.ply","w") as day:
#     # add the header for the 3 point clouds files that are being generated
#     # day.write("ply\nformat ascii 1.0\nelement vertex "+str(n_lines)+"\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header")

#     print("Colorizing point cloud...")
#     for line in tqdm(file, total=n_lines):
#         line = line.split()

#         coords3d = (line[0],line[1],line[2])
#         coords2d = (line[3],line[4])
#         color_mapping[coords2d] = [x for x in coords3d]

#         try:
#             if line[5] == '0':
#                 rgbDay = pix_rgb_day[0][int(line[3]),int(line[4])]
#                 #r = 0
#             if line[5] == '1':
#                 rgbDay = pix_rgb_day[1][int(line[3]),int(line[4])]
#                 #g = 0
#             if line[5] == '2':
#                 rgbDay = pix_rgb_day[2][int(line[3]),int(line[4])]
#                 #b = 0
                
#             day.write("\n"+line[0]+" "+line[1]+" "+line[2]+" "+str(rgbDay[0])+" "+str(rgbDay[1])+" "+str(rgbDay[2]))
#         except:
#             pass