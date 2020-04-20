import numpy as np
import cv2
import defs



cameraFeed= False
videoPath = 'lanevid.mp4'
cameraNo= 1
frameWidth= 640
frameHeight = 480


intialTracbarVals = [42,63,14,87]   


cap = cv2.VideoCapture(videoPath)
count=0
noOfArrayValues =10
global arrayCurve, arrayCounter
arrayCounter=0
arrayCurve = np.zeros([noOfArrayValues])
myVals=[]
defs.initializeTrackbars(intialTracbarVals)


while True:

    success, img = cap.read()
    try:
        if cameraFeed== False:img = cv2.resize(img, (frameWidth, frameHeight), None)
    except Exception as e:
        print(str(e))
    imgWarpPoints = img.copy()
    imgFinal = img.copy()
    imgCanny = img.copy()

    imgUndis = defs.undistort(img)
    imgThres,imgCanny,imgColor = defs.thresholding(imgUndis)
    src = defs.valTrackbars()
    imgWarp = defs.perspective_warp(imgThres, dst_size=(frameWidth, frameHeight), src=src)
    imgWarpPoints = defs.drawPoints(imgWarpPoints, src)
    imgSliding, curves, lanes, ploty = defs.sliding_window(imgWarp, draw_windows=True)

    try:
        curverad =defs.get_curve(imgFinal, curves[0], curves[1])
        lane_curve = np.mean([curverad[0], curverad[1]])
        imgFinal = defs.draw_lanes(img, curves[0], curves[1],frameWidth,frameHeight,src=src)

        currentCurve = lane_curve // 50
        if  int(np.sum(arrayCurve)) == 0:averageCurve = currentCurve
        else:
            averageCurve = np.sum(arrayCurve) // arrayCurve.shape[0]
        if abs(averageCurve-currentCurve) >200: arrayCurve[arrayCounter] = averageCurve
        else :arrayCurve[arrayCounter] = currentCurve
        arrayCounter +=1
        if arrayCounter >=noOfArrayValues : arrayCounter=0
        cv2.putText(imgFinal, str(int(averageCurve)), (frameWidth//2-70, 70), cv2.FONT_HERSHEY_DUPLEX, 1.75, (0, 0, 255), 2, cv2.LINE_AA)

    except:
        lane_curve=00
        pass

    imgFinal= defs.drawLines(imgFinal,lane_curve)


    imgThres = cv2.cvtColor(imgThres,cv2.COLOR_GRAY2BGR)
    imgBlank = np.zeros_like(img)
    imgStacked = defs.stackImages(0.7, ([img,imgUndis,imgWarpPoints],
                                         [imgColor, imgCanny, imgThres],
                                         [imgWarp,imgSliding,imgFinal]
                                         ))

    cv2.imshow("PipeLine",imgStacked)
    cv2.imshow("Result", imgFinal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
