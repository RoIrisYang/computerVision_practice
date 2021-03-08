import cv2
import numpy as np
import imutils

video = "video/clap.mp4"
cap = cv2.VideoCapture(video)
videoLen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

mog = cv2.bgsegm.createBackgroundSubtractorMOG()
mog2 = cv2.createBackgroundSubtractorMOG2()
knn = cv2.createBackgroundSubtractorKNN()
gmg = cv2.bgsegm_BackgroundSubtractorGMG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

if __name__ == "__main__":
    out = cv2.VideoWriter("BGSubstract_compare.avi",  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          30, (int(width) * 2, int(height * 3)))

    preFrame = None
    for i in range(0, videoLen):
        print(i, "/", videoLen)
        ret, frame = cap.read()
        curFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if preFrame is None:
            preFrame = curFrame

        fgmask_diff = cv2.absdiff(curFrame, preFrame)
        preFrame = curFrame

        """
        fgmask_mog = cv2.morphologyEx(mog.apply(curFrame), cv2.MORPH_OPEN, None)
        fgmask_mog2 = cv2.morphologyEx(mog2.apply(curFrame), cv2.MORPH_OPEN, None)
        fgmask_knn = cv2.morphologyEx(knn.apply(curFrame), cv2.MORPH_OPEN, None)
        fgmask_gmg = cv2.morphologyEx(gmg.apply(curFrame), cv2.MORPH_OPEN, None)
        """
        fgmask_mog = mog.apply(curFrame)
        fgmask_mog2 = mog2.apply(curFrame)
        fgmask_knn = knn.apply(curFrame)
        print("get a")
        # fgmask_gmg = gmg.apply(curFrame)
        print("get b")
        # fgmask_gmg = cv2.morphologyEx(fgmask_gmg, cv2.MORPH_OPEN, kernel)
        print("get c")
        boundColor = (255, 255, 255)
        fontSize = 1
        locX = 0
        locY = 50
        b = np.zeros(fgmask_mog.shape[:2], dtype="uint8")
        r = np.zeros(fgmask_mog.shape[:2], dtype="uint8")

        fgmask_diff_rgb = cv2.merge([b, fgmask_diff, r])
        cv2.putText(fgmask_diff_rgb, "cv2.absdiff", (locX, locY), cv2.FONT_HERSHEY_COMPLEX, fontSize, boundColor, 2)

        fgmask_mog_rgb = cv2.merge([b, fgmask_mog, r])
        cv2.putText(fgmask_mog_rgb, "MOG", (locX, locY), cv2.FONT_HERSHEY_COMPLEX, fontSize, boundColor, 2)

        fgmask_mog2_rgb = cv2.merge([b, fgmask_mog2, r])
        cv2.putText(fgmask_mog2_rgb, "MOG2", (locX, locY), cv2.FONT_HERSHEY_COMPLEX, fontSize, boundColor, 2)

        fgmask_knn_rgb = cv2.merge([b, fgmask_knn, r])
        cv2.putText(fgmask_knn_rgb, "KNN", (locX, locY), cv2.FONT_HERSHEY_COMPLEX, fontSize, boundColor, 2)

        # gmg cannot work but I don't know why
        fgmask_gmg_rgb = cv2.merge([b, fgmask_knn, r])
        cv2.putText(fgmask_gmg_rgb, "GMG", (locX, locY), cv2.FONT_HERSHEY_COMPLEX, fontSize, boundColor, 2)

        cv2.putText(frame, "Original", (locX, locY), cv2.FONT_HERSHEY_COMPLEX, fontSize, boundColor, 2)

        print(frame.shape, fgmask_diff_rgb.shape)
        combined1 = np.hstack((frame, fgmask_mog_rgb))
        combined2 = np.hstack((fgmask_mog2_rgb, fgmask_diff_rgb))
        combined3 = np.hstack((fgmask_knn_rgb, fgmask_gmg_rgb))
        combined = np.vstack((combined1, combined2, combined3))
        print(combined.shape)
        # combined = imutils.resize(combined, width=1980)
        # cv2.imshow("Combined", imutils.resize(combined, width=600))
        out.write(combined)

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
