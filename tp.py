import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imutils
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import webcolors 

Tk().withdraw() 
image_path = askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]) #selecting the image

if not image_path:
    print("No image selected. Exiting...")
    exit()

img = cv2.imread(image_path)
org_img = img.copy()
print('Org image shape --> ', img.shape)

img = imutils.resize(img, height=200)
print('After resizing shape --> ', img.shape)

# Flattening the image for KMeans clustering
flat_img = np.reshape(img, (-1, 3))
print('After Flattening shape --> ', flat_img.shape)

# using KMeans clustering
clusters = 5  
kmeans = KMeans(n_clusters=clusters, random_state=0)
kmeans.fit(flat_img)

dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')
percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]
p_and_c = zip(percentages, dominant_colors)
p_and_c = sorted(p_and_c, reverse=True)

def bgr_to_hex(bgr):
    return '#{:02x}{:02x}{:02x}'.format(bgr[2], bgr[1], bgr[0]) #bgr to hex format

block = np.ones((50, 50, 3), dtype='uint') # here we have creating a block with most dominant color 
plt.figure(figsize=(12, 8))

for i in range(clusters):
    plt.subplot(2, clusters, i + 1)
    block[:] = p_and_c[i][1][::-1]  # Converting the BGR to RGB for using plt function so that its easy to plot 
    plt.imshow(block)
    plt.xticks([])
    plt.yticks([])
    
    # Converting to hex color and obtaining their name
    hex_color = bgr_to_hex(p_and_c[i][1])
    try:
        color_name = webcolors.rgb_to_name(tuple(p_and_c[i][1].astype(int)))
    except ValueError:
        color_name = "Unknown"  # In case the color is not named in the library
    
    plt.xlabel(f'{color_name}\n{hex_color}\n{round(p_and_c[i][0] * 100, 2)}%')

    color = p_and_c[i][1]
    dist = np.linalg.norm(flat_img - color, axis=1)
    mask = dist < 50  # Adjust this threshold for more/less strict color matching

    # plotting histogram for the dominant color
    plt.subplot(2, clusters, i + clusters + 1)
    color_pixels = flat_img[mask]
    color_hist = cv2.calcHist([color_pixels], [0], None, [256], [0, 256])  # Histogram for R, G, or B channel

    plt.plot(color_hist, color='k')
    plt.xlim([0, 256])
    plt.title(f'Hist of Color {i+1}')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')


bar = np.ones((50, 500, 3), dtype='uint') # creating a bar to show color composition
plt.figure(figsize=(12, 8))
plt.title('Proportions of colors in the image')

# here we gonna make color composition bars
start = 0
i = 1
for p, c in p_and_c:
    end = start + int(p * bar.shape[1])
    if i == clusters:
        bar[:, start:] = c[::-1]
    else:
        bar[:, start:end] = c[::-1]
    start = end
    i += 1

plt.imshow(bar)
plt.xticks([])
plt.yticks([])

# Resize original image for display
rows = 1000
cols = int((org_img.shape[0] / org_img.shape[1]) * rows)
img = cv2.resize(org_img, dsize=(rows, cols), interpolation=cv2.INTER_LINEAR)

copy = img.copy()

cv2.rectangle(copy, (rows // 2 - 250, cols // 2 - 90), (rows // 2 + 250, cols // 2 + 110), (255, 255, 255), -1)
final = cv2.addWeighted(img, 0.1, copy, 0.9, 0)
cv2.putText(final, 'Most Dominant Colors in the Image', (rows // 2 - 230, cols // 2 - 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

# all we gotta do is display the most dominant color 
start = rows // 2 - 220
for i in range(clusters):
    end = start + 70
    final[cols // 2:cols // 2 + 70, start:end] = p_and_c[i][1]
    hex_color = bgr_to_hex(p_and_c[i][1])
    try:
        color_name = webcolors.rgb_to_name(tuple(p_and_c[i][1].astype(int)))
    except ValueError:
        color_name = "Unknown"  # In case the color is not named in the library but like every color is unknown lmaoo
    cv2.putText(final, f'{i+1}: {color_name}\n{hex_color}', (start + 10, cols // 2 + 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    start = end + 20

# displaying result
plt.show()
cv2.imshow('img', final)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output.png', final)
