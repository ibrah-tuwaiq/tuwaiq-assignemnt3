
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("img/cancer.png")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

blue_channel = img[:, :, 0]

blur = cv2.GaussianBlur(blue_channel, (5, 5), 0)

_, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
dilate = cv2.morphologyEx(opening, cv2.MORPH_DILATE, kernel, iterations=2)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(rgb)

plt.subplot(2, 2, 2)
plt.imshow(blue_channel, cmap="gray")

plt.subplot(2, 2, 3)
plt.imshow(thresh, cmap="gray")

plt.subplot(2, 2, 4)
plt.imshow(dilate, cmap="gray")

plt.tight_layout()
plt.show()

output_file = "segmented_cancer_cells.png"
cv2.imwrite(output_file, dilate)

