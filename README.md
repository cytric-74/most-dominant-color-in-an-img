# Dominant Color Detection in Images using K-Means Clustering

This project allows users to **analyze and visualize the dominant colors** in an image using **K-Means clustering**. The script extracts the top colors, visualizes their percentages, shows their histograms, and overlays them on the original image with proper color names and hex codes.

---

## Why did I make this project??

Colors carry **semantic and psychological meaning**, influence **human perception**, and are often the first features noticed in images. Automatically detecting dominant colors has applications in:

- **Image indexing and retrieval**
- **Digital art and design tools**
- **E-commerce product color classification**
- **Interior design & fashion tech**
- **AI aesthetics analysis**
  
This project combines **unsupervised machine learning (K-Means clustering)** with **image processing** to quantify and visualize the color composition of an image.

---

## Input vs Output Example

| Original Image | Output Image |
|----------------|--------------|
| ![original](docs/sample_input.jpg) | ![output](docs/output.png) |

---

## 📌 Features

- GUI-based file selector using `Tkinter`
- Resizes image for faster computation
- Applies **KMeans clustering** on pixel data
- Displays:
  - Top dominant colors in blocks with their names and hex codes
  - Individual color histograms
  - A proportion bar representing color distribution
  - Overlays on the original image showing dominant colors

---

## Scientific Background

### K-Means Clustering

K-Means is an **unsupervised machine learning algorithm** that partitions data into `k` clusters based on similarity.

#### Mathematically:
- Given: data points \( X = \{x_1, x_2, ..., x_n\} \)
- Objective: Minimize the within-cluster sum of squares (WCSS)

\[
\text{WCSS} = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2
\]

Where \( \mu_i \) is the mean of cluster \( C_i \).

In our case, each pixel \( x_i = [B, G, R] \) is treated as a point in 3D color space.

---

### Color Naming via Webcolors

After clustering, each centroid (dominant color) is matched to a **closest named color** using the `webcolors` library. If a name is not found (i.e., not in CSS3 color set), it is labeled as **Unknown**.

---

### 📊 Histogram Visualization

We compute histograms for the matched pixels per color cluster using OpenCV's `cv2.calcHist`. This shows the **intensity distribution** of each dominant color.



---

## Tech Stack

| Library         | Purpose                              |
|-----------------|--------------------------------------|
| `OpenCV`        | Image reading, processing & drawing  |
| `NumPy`         | Array handling and math operations   |
| `matplotlib`    | Plotting colors and histograms       |
| `scikit-learn`  | KMeans clustering                    |
| `webcolors`     | Mapping RGB to named colors          |
| `tkinter`       | GUI File Dialog                      |
| `imutils`       | Image resizing utilities             |

---

## How It Works

![KMeans Clustering Diagram](https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif)

```python
flat_img = np.reshape(img, (-1, 3))
kmeans = KMeans(n_clusters=5)
kmeans.fit(flat_img)
