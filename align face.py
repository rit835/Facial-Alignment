import cv2
import dlib
from imutils import face_utils
import numpy as np
import imutils
from imutils.video import  WebcamVideoStream

class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        if (len(shape)==68):
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_5_IDXS["right_eye"]
        
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        return angle

file =  WebcamVideoStream(src=0).start()
detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
face = FaceAligner(predictor, desiredFaceWidth=600)
while True:
    frame = file.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect = detector(gray)
    if(len(detect)>0):
        for det in detect:
            shape = predictor(gray,det)
            shape = face_utils.shape_to_np(shape)
            align = face.align(frame, gray, det)
            (x, y, w, h) = face_utils.rect_to_bb(det)
            if(-360.0 <= align <= -353.0) or (-7.0 <= align <= 0):
                cv2.rectangle(frame,(det.left(), det.top()), (det.right(), det.bottom()),(0,255,0),3)
                cv2.putText(frame,"FACE ALIGNED",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                for (x,y) in shape:
                    cv2.circle(frame,(x,y),1,(0,255,0),1)
                
            else:
                cv2.rectangle(frame,(det.left(), det.top()), (det.right(), det.bottom()),(0,0,255),3)
                cv2.putText(frame,"PLEASE ALIGN YOUR FACE!",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    else:
        cv2.putText(frame, "NO FACE PRESENT",(20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        file.stop()
        break
file.stream.release()
cv2.destroyAllWindows()