import cv2
import face_recognition
 
imgElon = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Bill gates.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
 
faceLoc = face_recognition.face_locations(imgElon)&amp;#91;0]
encodeElon = face_recognition.face_encodings(imgElon)&amp;#91;0]
cv2.rectangle(imgElon,(faceLoc&amp;#91;3],faceLoc&amp;#91;0]),(faceLoc&amp;#91;1],faceLoc&amp;#91;2]),(255,0,255),2)
 
faceLocTest = face_recognition.face_locations(imgTest)&amp;#91;0]
encodeTest = face_recognition.face_encodings(imgTest)&amp;#91;0]
cv2.rectangle(imgTest,(faceLocTest&amp;#91;3],faceLocTest&amp;#91;0]),(faceLocTest&amp;#91;1],faceLocTest&amp;#91;2]),(255,0,255),2)
 
results = face_recognition.compare_faces(&amp;#91;encodeElon],encodeTest)
faceDis = face_recognition.face_distance(&amp;#91;encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis&amp;#91;0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)