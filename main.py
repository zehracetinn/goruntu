import cv2
import matplotlib.pyplot as plt
import numpy as np

# görüntü oku
img_bgr = cv2.imread("test.jpg")

print("Shape (H, W, C):", img_bgr.shape)
print("Dtype:", img_bgr.dtype)

# BGR → RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# GRAYSCALE
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# pixel istatistikleri
print("Minimum pixel:", img_gray.min())
print("Maximum pixel:", img_gray.max())
print("Mean brightness:", img_gray.mean())

# RGB ve grayscale göster
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("RGB Image")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Grayscale")
plt.imshow(img_gray, cmap="gray")
plt.axis("off")

plt.show()


# HISTOGRAM
plt.figure()
plt.title("Histogram")
plt.hist(img_gray.ravel(), bins=256)
plt.xlabel("Pixel Value")
plt.ylabel("Pixel Count")
plt.show()


# PARLAKLIK ARTIRMA
bright = cv2.add(img_gray, 50)

plt.figure()
plt.imshow(bright, cmap="gray", vmin=0, vmax=255)
plt.title("Brightness +50")
plt.axis("off")
plt.show()
print("Original mean:", img_gray.mean())
print("Bright mean:", bright.mean())


gaussian = cv2.GaussianBlur(img_gray, (5,5), 0)

plt.imshow(gaussian, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Sobel X")
plt.imshow(sobelx, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Sobel Y")
plt.imshow(sobely, cmap="gray")
plt.axis("off")

plt.show()


edge_strength = np.sqrt(sobelx**2 + sobely**2)

plt.imshow(edge_strength, cmap="gray")
plt.title("Edge Strength")
plt.axis("off")
plt.show()
_, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

plt.imshow(thresh, cmap="gray")
plt.title("Binary Image")
plt.axis("off")
plt.show()




_, otsu = cv2.threshold(
    img_gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

plt.imshow(otsu, cmap="gray")
plt.title("Otsu Threshold")
plt.axis("off")
plt.show()


adaptive = cv2.adaptiveThreshold(
    img_gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

plt.imshow(adaptive, cmap="gray")
plt.title("Adaptive Threshold")
plt.axis("off")
plt.show()

contours, hierarchy = cv2.findContours(
    otsu,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Contour sayısı:", len(contours))
img_copy = img_rgb.copy()

cv2.drawContours(
    img_copy,
    contours,
    -1,
    (255,0,0),
    2
)

plt.imshow(img_copy)
plt.title("Contours")
plt.axis("off")
plt.show()
img_copy = img_rgb.copy()

for c in contours:

    area = cv2.contourArea(c)

    print("Alan:", area)

    if area > 1000:

        # contour çiz
        cv2.drawContours(img_copy, [c], -1, (0,255,0), 3)

        # bounding box
        x,y,w,h = cv2.boundingRect(c)

        cv2.rectangle(
            img_copy,
            (x,y),
            (x+w,y+h),
            (255,0,0),
            3
        )

plt.imshow(img_copy)
plt.title("Filtered Objects")
plt.axis("off")
plt.show()

edges = cv2.Canny(img_gray, 100, 200)


plt.imshow(edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")
plt.show()
kernel = np.ones((5,5), np.uint8)

opening = cv2.morphologyEx(
    otsu,
    cv2.MORPH_OPEN,
    kernel
)

plt.imshow(opening, cmap="gray")
plt.title("Opening Result")
plt.axis("off")
plt.show()
contours, _ = cv2.findContours(
    opening,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Yeni contour sayısı:", len(contours))
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

plt.imshow(hsv[:,:,0], cmap="gray")
plt.title("Hue Channel")
plt.axis("off")
plt.show()
lower = np.array([140, 50, 50])
upper = np.array([170, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

plt.imshow(mask, cmap="gray")
plt.title("Pink Mask")
plt.axis("off")
plt.show()
gray = np.float32(img_gray)

corners = cv2.cornerHarris(
    gray,
    2,
    3,
    0.04
)

corners = cv2.dilate(corners, None)

img_corner = img_rgb.copy()

img_corner[corners > 0.01 * corners.max()] = [255,0,0]

plt.imshow(img_corner)
plt.title("Harris Corners")
plt.axis("off")
plt.show()

#orb

orb = cv2.ORB_create()

kp, des = orb.detectAndCompute(img_gray, None)

img_orb = cv2.drawKeypoints(
    img_rgb,
    kp,
    None,
    color=(0,255,0),
    flags=0
)

print("Keypoint sayısı:", len(kp))

plt.imshow(img_orb)
plt.title("ORB Features")
plt.axis("off")
plt.show()
img2 = cv2.imread("test2.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img_gray, None)
kp2, des2 = orb.detectAndCompute(img2_gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

img_matches = cv2.drawMatches(
    img_rgb, kp1,
    img2, kp2,
    matches[:20],
    None,
    flags=2
)

plt.imshow(img_matches)
plt.title("Feature Matching")
plt.axis("off")
plt.show()

pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

H, mask = cv2.findHomography(
    pts1,
    pts2,
    cv2.RANSAC,
    5.0
)

print("Homography Matrix:")
print(H)

hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

lower = np.array([140,50,50])
upper = np.array([170,255,255])

mask = cv2.inRange(hsv, lower, upper)

kernel = np.ones((5,5),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


edges = cv2.Canny(img_gray, 100, 200)

lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi/180,
    100,
    minLineLength=100,
    maxLineGap=10
)

img_lines = img_rgb.copy()

if lines is not None:
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img_lines,(x1,y1),(x2,y2),(255,0,0),2)

plt.imshow(img_lines)
plt.title("Detected Lines")
plt.axis("off")
plt.show()

circles = cv2.HoughCircles(
    img_gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=100,
    param1=100,
    param2=30,
    minRadius=20,
    maxRadius=200
)

img_circle = img_rgb.copy()

if circles is not None:

    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:

        x,y,r = i

        cv2.circle(img_circle,(x,y),r,(255,0,0),3)

        cv2.circle(img_circle,(x,y),2,(0,255,0),3)

plt.imshow(img_circle)
plt.title("Detected Circles")
plt.axis("off")
plt.show()

# blur
blur = cv2.GaussianBlur(img_gray, (7,7), 0)

# edge detection
edges = cv2.Canny(blur, 50, 150)

# hough line transform
lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi/180,
    threshold=150,
    minLineLength=200,
    maxLineGap=30
)

img_lines = img_rgb.copy()

if lines is not None:
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img_lines,(x1,y1),(x2,y2),(255,0,0),3)

plt.imshow(img_lines)
plt.title("Better Line Detection")
plt.axis("off")
plt.show()

lsd = cv2.createLineSegmentDetector(0)

lines, _, _, _ = lsd.detect(img_gray)

img_lines = img_rgb.copy()

if lines is not None:
    for line in lines:
        x1,y1,x2,y2 = map(int, line[0])
        cv2.line(img_lines,(x1,y1),(x2,y2),(255,0,0),2)

plt.imshow(img_lines)
plt.title("LSD Line Detection")
plt.axis("off")
plt.show()