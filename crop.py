import cv2
import matplotlib.pyplot as plt
img = cv2.imread("3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
dst = cv2.Canny(gray, 0, 150)
blured = cv2.blur(dst, (5,5), 0)    
MIN_CONTOUR_AREA=200
img_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
Contours,imgContours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for contour in Contours:
    if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
        [X, Y, W, H] = cv2.boundingRect(contour)
        box=cv2.rectangle(img, (X, Y), (X + W, Y + H), (0,0,255), 2)

cropped_image = img[Y:Y+H, X:X+W]
print([X,Y,W,H])
plt.imshow(cropped_image)
cv2.imwrite('contour1.png', cropped_image)