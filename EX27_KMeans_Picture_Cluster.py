from skimage import color
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
import PIL.Image as image
import matplotlib.image as mpimg

def load_data(file_path):
    f = open(file_path, 'rb')
    data = []
    img = image.open(f)
    width, height = img.size
    for x in range(width):
        for y in range(height):
            c1,c2,c3 = img.getpixel((x,y))
            data.append([c1,c2,c3])
            
    f.close()
    mm = preprocessing.MinMaxScaler()
    data = mm.fit_transform(data)
    return np.mat(data), width, height

img, width, height = load_data("C:/Users/qye/Python/Learning/WeChat.jpg")

kmeans = KMeans(n_clusters=2)
kmeans.fit(img)
label = kmeans.predict(img)
label = label.reshape([width, height])
pic_mark = image.new("L", (width,height))
for x in range(width):
    for y in range(height):
        pic_mark.putpixel((x,y), int(256/(label[x][y]+1))-1)
pic_mark.save("C:/Users/qye/Python/Learning/WeChat_mark.jpg", "JPEG")

# 将聚类标识矩阵转化为不同颜色的矩阵
kmeans_color = KMeans(n_clusters=16)
label_color = kmeans_color.fit_predict(img)
label_color = label_color.reshape([width, height])
label_new = label_color
label_color = (color.label2rgb(label_color)*255).astype(np.uint8)
label_color = label_color.transpose(1,0,2)
images = image.fromarray(label_color)
images.save("C:/Users/qye/Python/Learning/WeChat_mark_color.jpg")

# 创建个新图像img，用来保存图像聚类压缩后的结果
new_img = image.new('RGB', (width, height))
for x in range(width):
    for y in range(height):
        c1 = kmeans_color.cluster_centers_[label_new[x,y],0]
        c2 = kmeans_color.cluster_centers_[label_new[x,y],1]
        c3 = kmeans_color.cluster_centers_[label_new[x,y],2]
        new_img.putpixel((x,y),(int(c1*256)-1, int(c2*256)-1,int(c3*256)-1))
        
new_img.save("C:/Users/qye/Python/Learning/WeChat_new_color.jpg")
print("done")
