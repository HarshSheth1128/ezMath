# import the necessary packages
import imutils
import cv2
import numpy as np
import random as rng

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("IMG_0268.png", 0)

# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
ret, th2 = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
kernel = np.ones((2, 2), np.uint8)
# th2 = cv2.dilate(th2, kernel, iterations=1)
# kernel2 = np.ones((4, 4), np.uint8)
# th2 = cv2.erode(th2, kernel, iterations=3)
# th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
H, W = th2.shape[:2]


image = np.zeros((H, W, 3), np.uint8)

cv2.drawContours(image, contours, -1, (0, 0, 255), 3)


hist = cv2.reduce(th2, 1, cv2.REDUCE_AVG).reshape(-1)

th = 255
uppers = np.array([y for y in range(H - 1) if hist[y] >= th and hist[y + 1] < th])
lowers = np.array([y for y in range(H - 1) if hist[y] < th and hist[y + 1] >= th])

print(uppers)
print(lowers)

# # Draw lines
# for line in uppers:
#     cv2.line(th2, (0, line), (W, line), (0, 0, 0), 2)

# for line in lowers:
#     cv2.line(th2, (0, line), (W, line), (0, 0, 0), 2)

# Get the bounding boxes and sort them
boundingBoxes = [cv2.boundingRect(c) for c in contours]
contours, boundingBoxes = zip(
    *sorted(zip(contours, boundingBoxes), key=lambda b: b[1][0])
)

print(len(contours))
print(boundingBoxes)


def isBoxInBox(box1, box2):
    if (
        box2[0] <= box1[0] <= box2[0] + box2[2] and
        box2[0] <= box1[0] + box1[2] <= box2[0] + box2[2] and
        box2[1] <= box1[1] <= box2[1] + box2[3] and
        box2[1] <= box1[1] + box1[3] <= box2[1] + box2[3]
    ):
        return True
    else:
        return False


def removeInternalBoundingBoxesAndContours(boundingBoxes, contours):
    # Remove the parent box
    boundingBoxes = list(boundingBoxes)[1:]
    contours = contours[1:]
    for i, box in enumerate(boundingBoxes):
        for j, box2 in enumerate(boundingBoxes):
            if i != j and isBoxInBox(box, box2):
                print(str(box) + " in " + str(box2))
                boundingBoxes[i] = (0, 0, 0, 0)
                break

    newContours = [cnt for i, cnt in enumerate(contours) if boundingBoxes[i] != (0, 0, 0, 0)]
    return newContours, [*filter(lambda el: el != (0, 0, 0, 0), boundingBoxes)]


filteredContours = []
contours, boundingBoxes = removeInternalBoundingBoxesAndContours(boundingBoxes, contours)

print(boundingBoxes)
print(len(contours))

symbols = []

for i in range(len(boundingBoxes)):
    color = (255, 255, 255)
    # cv2.rectangle(image, (int(boundingBoxes[i][0]), int(boundingBoxes[i][1])),
    #               (int(boundingBoxes[i][0] + boundingBoxes[i][2]), int(boundingBoxes[i][1] + boundingBoxes[i][3])), color, 2)
    symbol = th2[boundingBoxes[i][1]:boundingBoxes[i][1] + boundingBoxes[i][3], boundingBoxes[i][0]: boundingBoxes[i][0] + boundingBoxes[i][2]]

    # Add padding to the image if needed
    H, W = symbol.shape[:2]

    if H > W:
        paddingNeeded = H - W
        symbol = cv2.copyMakeBorder(symbol, 10, 10, paddingNeeded + 10, paddingNeeded + 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    elif H < W:
        paddingNeeded = W - H
        symbol = cv2.copyMakeBorder(symbol, paddingNeeded + 10, paddingNeeded + 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        symbol = cv2.copyMakeBorder(symbol, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    symbol = cv2.bitwise_not(symbol)
    symbol = cv2.resize(symbol, (28, 28), interpolation=cv2.INTER_LINEAR)
    symbols.append(symbol)

for i, symbol in enumerate(symbols):
    cv2.imshow('symbol' + str(i), symbol)

# cv2.imshow('image', th2)
# cv2.waitKey(0)
# cv2.imshow('image contour', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
