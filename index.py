import cv2

img = cv2.imread("./images/anh_the.jpg")

net= cv2.dnn.readNetFromCaffe(
    "./modules/deploy.prototxt",
    "./modules/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

blod = cv2.dnn.blobFromImage(img,1.0,(300,300),(104,177,123),swapRB = False)
net.setInput(blod)
faces = net.forward()
print(faces.shape)
print(faces[0,0,0,])

h = img.shape[0]
w = img.shape[1]

for i in range(0, faces.shape[2]):
    confidence = faces[0,0,i,2]
    if confidence > 0.5:
        print(faces[0,0,i,3:7])
        startx = int(faces[0,0,i,3] * w)
        starty = int(faces[0, 0, i, 4] * h)
        endx = int(faces[0,0,i,5] * w)
        endy = int(faces[0,0,i,6] * h)
        print(startx,starty,endx,endy)

        cv2.rectangle(img,(startx,starty),(endx,endy),(0,255,0),1)

        text = "Face: {:.2f}%".format(confidence*100)
        cv2.putText(img,text,(startx,starty-10),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)

cv2.imshow("Anh goc",img)
cv2.waitKey()
cv2.destroyAllWindows()