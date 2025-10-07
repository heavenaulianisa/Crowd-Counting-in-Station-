import numpy as np
import cv2
import time

cap = cv2.VideoCapture(r"D:\Comvis\basic-code-detection\kereta1.mp4") ##masukan video/camera webcam

net = cv2.dnn.readNetFromONNX(r"D:\Comvis\basic-code-detection\best.onnx")  ##masukan model onnx

classes = ["head", "helmet", "person"]


while True:
    window, image = cap.read()
    if window == False:
        break
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    # mengolah data setiap 30 frame
    if pos_frame%30 == 0:
        print(str(pos_frame)+" frames")
        img = image.copy()
   
        if img is None:
            break
        
        start = time.time()
        blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
        net.setInput(blob)
        detections = net.forward()[0]

        classes_ids = []
        confidences = []
        boxes = []
        rows = detections.shape[0]

        img_width, img_height = img.shape[1], img.shape[0]
        x_scale = img_width/640
        y_scale = img_height/640

        for i in range(rows):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.2:
                classes_score = row[5:]
                ind = np.argmax(classes_score)
                if classes_score[ind] > 0.2:
                    classes_ids.append(ind)
                    confidences.append(confidence)
                    cx, cy, w, h = row[:4]
                    x1 = int((cx- w/2)*x_scale)
                    y1 = int((cy-h/2)*y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([x1,y1,width,height])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes,confidences,0.2,0.2)
        cv2.rectangle(img,(0, 0),(180, 45),(0,0,0),-1)


        id_orang = 1
        for i in indices:
            x1,y1,w,h = boxes[i]
            label = classes[classes_ids[i]]
            conf = confidences[i]
            text = label + "{:.2f}".format(conf)
            cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),2)

            cv2.putText(img, f"ID {id_orang}, {conf:.2f}", (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.6, (71,150,255),2)
            id_orang += 1

        end = time.time()

        img = cv2.resize(img, (640,720))

        image = cv2.resize(image, (640,720))
        im_h = cv2.hconcat([image, img])
        cv2.imshow("VIDEO",im_h)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break


