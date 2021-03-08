import cv2
import numpy as np

imageDir = "image/"
carImage = cv2.imread(imageDir + "car_complexBG.jpg", 0)
carImage_BG = cv2.imread(imageDir + "car_complexBG_empty.jpg", 0)
mdImage = cv2.imread(imageDir + "md_grayBG.jpg", 0)
mdImage_BG = cv2.imread(imageDir + "md_grayBG_empty.jpg", 0)


def segment_absDiff():
    diff_car = cv2.absdiff(carImage, carImage_BG)
    cv2.imwrite("car_absDiff.jpg", diff_car)

    diff_md = cv2.absdiff(mdImage, mdImage_BG)
    cv2.imwrite("md_absDiff.jpg", diff_md)

    return diff_car, diff_md


if __name__ == "__main__":
    absDiff_car, absDiff_md = segment_absDiff()

    # Image Thresholding
    st_car, binaryThr_car = cv2.threshold(absDiff_car, 127, 255, cv2.THRESH_BINARY)
    adpativeGaussianThr_md = cv2.adaptiveThreshold(absDiff_md, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    cv2.imwrite("car_binaryThr.jpg", binaryThr_car)
    cv2.imwrite("md_adaptiveGaussianThr.jpg", adpativeGaussianThr_md)

    # erode: shrinking noises
    kernel = np.ones((5, 5), np.uint8)
    erode_car_3x3 = cv2.erode(binaryThr_car, None)  # kernel = None: default (3x3)
    erode_car_5x5 = cv2.erode(binaryThr_car, kernel)
    cv2.imwrite("car_erode_3x3.jpg", erode_car_3x3)
    cv2.imwrite("car_erode_5x5.jpg", erode_car_5x5)

    # erode + dilate
    fill_md = cv2.erode(adpativeGaussianThr_md, None)
    fill_md = cv2.dilate(fill_md, None)
    fill_md = cv2.erode(fill_md, None)
    cv2.imwrite("md_fill.jpg", fill_md)

    # find obj contours
    contours, status_con = cv2.findContours(binaryThr_car, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = binaryThr_car.copy()
    contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        print(x, y, w, h)
        cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite("car_contour.jpg", contour_image)
