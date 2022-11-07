import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
  img = cv2.imread(filename)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img)
  #plt.axes('off')
  plt.show()
  return img

filename = "/content/Friends.jpg"
img = read_file(filename)

org_img = np.copy(img)

def edge_mask(img, line_size, blur_value):
  """
  input:Input scale image
  output:Edges
  """
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  glay_blur = cv2.medianBlur(gray,blur_value)

  edges = cv2.adaptiveThreshold(glay_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
  return edges

line_size, blur_value = 9,7
edges = edge_mask(img, line_size, blur_value)

plt.imshow(edges, cmap = "gray")
plt.show()

def color_quantization(img, k):
  #transforme the image
  data = np.float32(img).reshape((-1,3))

  #determine criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

  ## implemeny k-means
  ret, label , center = cv2.kmeans(data,k, None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)

  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return  result

img = color_quantization(img, k=7)

plt.imshow(img)
plt.show()

#Reduce the noise

blurred = cv2.bilateralFilter(img, d = 7, sigmaColor =200,sigmaSpace =200)

plt.imshow(blurred )
plt.show()

def cartoon():
  c = cv2.bitwise_and(blurred , blurred , mask = edges)

  plt.imshow(c)
  plt.title("cartoon image")
  plt.show()

  plt.imshow(org_img)
  plt.title("Org image")
  plt.show(
      
cartoon()