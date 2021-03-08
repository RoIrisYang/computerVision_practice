import cv2

imageDir = "image/"
print(imageDir + "car_complexBG.jpg")
carImage = cv2.imread(imageDir + "car_complexBG.jpg")
carImage_BG = cv2.imread(imageDir + "car_complexBG_empty.jpg")
mdImage = cv2.imread(imageDir + "md_grayBG.jpg")
mdImage_BG = cv2.imread(imageDir + "md_grayBG_empty.jpg")


def segment_absDiff():
    diff_car = cv2.absdiff(carImage, carImage_BG)
    cv2.imwrite("car_absDiff.jpg", diff_car)

    diff_md = cv2.absdiff(mdImage, mdImage_BG)
    cv2.imwrite("md_absDiff.jpg", diff_md)


if __name__ == "__main__":
    segment_absDiff()
