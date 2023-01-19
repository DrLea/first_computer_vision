import pytesseract as tess
import cv2
import numpy as np

CONTOUR = (0, 255, 0), 4    #color(r,g,b), thickness
CORNER = (0, 25, 255), 25


def draw_border(max_contour, main_img):
        cv2.line(main_img, (max_contour[0][0][0], max_contour[0][0][1]), (max_contour[1][0][0], max_contour[1][0][1]), *CONTOUR)
        cv2.line(main_img, (max_contour[0][0][0], max_contour[0][0][1]), (max_contour[2][0][0], max_contour[2][0][1]), *CONTOUR)
        cv2.line(main_img, (max_contour[3][0][0], max_contour[3][0][1]), (max_contour[2][0][0], max_contour[2][0][1]), *CONTOUR)
        cv2.line(main_img, (max_contour[3][0][0], max_contour[3][0][1]), (max_contour[1][0][0], max_contour[1][0][1]), *CONTOUR)
        
source = r"1.jpg"
img = cv2.imread(source)
img = cv2.resize(img, (int(480*2), int(640*2)))
w, h = 480, 640
img_warp = img.copy()


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 1)
canny_img = cv2.Canny(blurred_img, 190, 190)
contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()
contour_img = cv2.drawContours(contour_img, contours, -1, *CONTOUR)
corner_img = img.copy()
max_area = 0
biggest = []

for i in contours :
    area = cv2.contourArea(i)
    if area > 500 :
        peri = cv2.arcLength(i, True)
        edges = cv2.approxPolyDP(i, 0.02*peri, True)
        if area > max_area and len(edges) == 4 :
            biggest = edges
            max_area = area

if len(biggest) != 0 :
    biggest = biggest.reshape((4, 2))
    max_contour = np.zeros((4, 1, 2), dtype= np.int32)
    add = biggest.sum(1)
    max_contour[0] = biggest[np.argmin(add)]
    max_contour[3] = biggest[np.argmax(add)]
    dif = np.diff(biggest, axis = 1)
    max_contour[1] = biggest[np.argmin(dif)]
    max_contour[2] = biggest[np.argmax(dif)]
    draw_border(max_contour, corner_img)
    corner_img = cv2.drawContours(corner_img, max_contour, -1, *CORNER)

    #rotating
    pts1 = np.float32(max_contour)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (w, h))


img = cv2.resize(img, (480, 640))
gray_img = cv2.resize(gray_img, (480, 640))
blurred_img = cv2.resize(blurred_img, (480, 640))
canny_img = cv2.resize(canny_img, (480, 640))
contour_img = cv2.resize(contour_img, (480, 640))
corner_img = cv2.resize(corner_img, (480, 640))


cv2.imshow("img", img)
cv2.imshow("gray_img", gray_img)
cv2.imshow("blurred_img", blurred_img)
cv2.imshow("canny_img", canny_img)
cv2.imshow("contour_img", contour_img)
cv2.imshow("corner_img", corner_img)
cv2.imshow("result", img_warp)
try:
    tess.pytesseract.tesseract_cmd = r"D:\tesseract\tesseract.exe"
    text = tess.image_to_string(img_warp).split('\n')
    print(text[-3], text[-2], text[-1], sep="\n")
except:
    print("unable to read the text")
cv2.waitKey(0)