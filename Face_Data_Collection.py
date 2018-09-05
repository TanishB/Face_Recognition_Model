import numpy as np
import cv2


#Initializing / Opening Camera
cap = cv2.VideoCapture(0)


'''There are sets for recognizing various things such as eyes , nose , face
    We here are concerned about recognising Frontal Face so we are loading
    the famous HAAR CASCADE set ...'''

face_cascade = cv2.CascadeClassifier('/Users/tanish/Downloads/DATA/haarcascade_frontalface_alt.xml')

skip = 0#skip variable used for skipping frames (just for saving storage)
#for Example if camera captures 60 frames in one second we set skip value so that we takes only some frames like 1 frame from 10

face_data = []#liat for collecting instances of a face

dataset_path = '/Users/tanish/Downloads/DATA/face_data/'#defining path where data will be stored

file_name = raw_input('Enter name of the person')

while True:
	
    ret , frame = cap.read()
    if ret == False:
    	continue
    #frame.shape = (480,640,3)

    #converting frame to grayscale becoz the function we will use requires gray scale image only
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    #there may be possibility that more than one person are in front of the camera so we will detect multiple faces
    faces = face_cascade.detectMultiScale(gray , 1.3 ,5)
    k = 1
    #faces = (x,y,w,h)

    #we are sorting multiple faces according to the area in decreasing order and will select the face with maximum area
    #print faces[1].shape
    faces = sorted(faces , key=lambda x: x[2]*x[3] , reverse = True)
    #2 and 3 parameter are of height and width multiplying which we will get area

    #updating skip
    skip += 1

    #faces list is sorted in decreasing order and we want only first element of the list i.e we use faces[:1]
    for face in faces[:1]:#taking first row

    	x , y , w , h = face
    	#as face consist of four parameters and we want to know their value

    	offset = 7

    	
    	#for making the box large 
    	#actually we have box's starting corner upper point as x,y and height as h and width as y
    	#as we want to make it large so--->>>
    	#for moving up y-offset
    	#for moving down y+h+offset
    	#for moving left x-offset
    	#for moving right x+h+offset
    	face_section = frame[y-offset : y+h+offset , x-offset : x+w+offset]
    	#just resizing for constant box size
    	face_section = cv2.resize(face_section , (100 , 100))


    	#we want to save data for every 10th frame to save our storage
    	if skip%10 == 0:
    		face_data.append(face_section)
    		print len(face_data)#no of faces instances


    	#display face ROI
    	cv2.imshow(str(k) , face_section)
    	k += 1


    	#draw a rectangle in the original image
    	#????????????????????
    	cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    
    cv2.imshow("Faces" , frame)#displays what is captured by camera

    if cv2.waitKey(1) & 0xFF == ord('q'):#bitwise and hexadecimal ff ##this statement is necessary for 64bit machine
    	break

face_data = np.asarray(face_data)
#face data shape. =(no_of_images ,100 ,100, 3)
face_data = face_data.reshape((face_data.shape[0] , -1))
print face_data.shape


#Saving data
np.save(dataset_path + file_name , face_data)
print "Dataset saved at {}".format(dataset_path + file_name + '.npy')

cv2.destroyAllWindows()
#SHORT SUMMARY::
'''camera on
   converting frame to gray
   detecting all face
   want biggest face
   applying offest
   capture every 10face
   converting faces list to array
   and saving it
 '''



