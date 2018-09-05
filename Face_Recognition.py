import numpy as np
import os
import cv2

#####-----KNN CODE-----#####

def distance(v1 , v2):
	return np.sqrt(((v1-v2)**2).sum())
	#it will take eucledian distance of 30k pixels(add it and will take square root)

def knn(train , test , k=5):

	dist =[]

	for i in range(train.shape[0]):


		ix = train[ i , : -1]
		iy = train[ i , -1]
		d = distance(test , ix)
		dist.append([d , iy])

	dk = sorted(dist , key=lambda x:x[0])[:k]
	labels = np.array(dk)[: , -1]
	output = np.unique(labels , return_counts = True)
	index = np.argmax(output[1])
	return output[0][index]

#####-----KNN CODE-----#####

#Initialize camera
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('/Users/tanish/Downloads/DATA/haarcascade_frontalface_alt.xml')

dataset_path = '/Users/tanish/Downloads/DATA/face_data/'

face_data = []
labels = []
class_id = 0

#Dataset Preparation

for x in os.listdir(dataset_path):#itwill iterate over all the files at that path one by one

	if x.endswith('.npy'):
		data_item = np.load(dataset_path + x)#1 person's all photos loading 
		face_data.append(data_item)#append is a list function

		target = class_id * np.ones((data_item.shape[0] , ))
		class_id += 1
		labels.append(target)

#?????????????why concatenating not asarray
face_data = np.concatenate(face_data , axis=0)
face_labels = np.concatenate(labels , axis=0).reshape((-1,1))
print face_data.shape
print face_labels.shape

trainset = np.concatenate((face_data , face_labels) , axis = 1)
print trainset.shape

names = {
	0 : 'Tanish' , 1 : 'Priyanshi' , 2 : 'Shubham' , 3 : 'Aayushi' , 4: 'Shivansh'
}

font = cv2.FONT_HERSHEY_SIMPLEX

while True:

	ret , frame = cap.read()
	if ret == False:
		continue

	gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for face in faces:#taking all faces not writing[:1]
		x , y , w , h = face
		offset = 7
		face_section = frame[y-offset : y+h+offset , x-offset : x+w+offset]
		face_section = cv2.resize(face_section , (100 , 100))
		out = knn(trainset , face_section.flatten())
		#faltten converts to 1 d array
		cv2.putText(frame , names[int(out)] , (x,y-10) , font , 1 , (255,0,0) , 2 , cv2.LINE_AA)
				
		cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,255,0) , 2)

	cv2.imshow("Faces" , frame)#to display big frame
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()