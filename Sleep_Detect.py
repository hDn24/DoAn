from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
from numpy import linalg as LA
import argparse
import imutils
from playsound import playsound
import time
from threading import Thread
import dlib
import os
import cv2
# import RPi.GPIO as GPIO

# Path file cảnh báo
wav_path = "D:\DS\DALTHT\SourceCode\Recording.m4a"

# Hàm phát âm thanh cảnh báo
def play_sound(path):
	os.system('aplay ' + path)

# Tính khoảng cách giữa 2 điểm
def distance(pA, pB):
	return LA.norm(pA - pB)

def eye_ratio(eye):
	# Khoảng cách chiều dọc giữa mí trên và mí dưới
	V1 = distance(eye[1], eye[5])
	V2 = distance(eye[2], eye[4])

	# Khoảng cách theo chiều ngang
	H = distance(eye[0], eye[3])

	# Tỷ lệ giữa chiều dọc và chiều ngang
	eye_ratio_val = (V1 + V2) / (2.0 * H)

	return eye_ratio_val

# define ngưỡng
eye_ratio_threshold = 0.25

# Số frame lớn nhất nhắm mắt liên tục
max_sleep_frames = 15

# Đếm số frame nhắm mắt
sleep_frames = 0

# kiểm tra cảnh báo
alarmed = False

# Khởi tạo module detect mắt và 68 point facial landmark
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Lấy vị trí cụm điểm lanmark cho mắt trái và phải
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Đọc video từ camera
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:

	# Đọc camera
	frame = vs.read()

	# Resize để tăng tốc độ xử lý
	frame = imutils.resize(frame, width=450)

	# Chuyển ảnh RGB sang GRAY
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect các mặt trong ảnh
	faces = face_detect.detectMultiScale(gray, scaleFactor=1.1,		minNeighbors=5, minSize=(100, 100))

	# Duyệt hết tất cả các mặt detect được
	for (x, y, w, h) in faces:

		# Lấy 1 khổi chữ nhật của khuôn mặt
		rect = dlib.rectangle(int(x), int(y), int(x + w),
			int(y + h))
		
		#vẽ 1 khối hcn lên khuôn mặt detect được
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

		# Nhận diện 68 point landmark
		landmark = landmark_detect(gray, rect)
		#Chuyển sang ndarray
		landmark = face_utils.shape_to_np(landmark)

		# lấy tọa đổ các cụm điểm lanmark ở 2 mắt
		leftEye = landmark[left_eye_start: left_eye_end]
		rightEye = landmark[right_eye_start: right_eye_end]

		# Tính tỉ lệ cho 2 mắt
		left_eye_ratio = eye_ratio(leftEye)
		right_eye_ratio = eye_ratio(rightEye)

		# Tính tỉ lệ trung bình
		eye_avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

		# vẽ đường bao quanh mắt
		left_eye_bound = cv2.convexHull(leftEye)
		right_eye_bound = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [left_eye_bound], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [right_eye_bound], -1, (0, 255, 0), 1)

		# Kiểm tra mắt nhắm hay mở
		if eye_avg_ratio < eye_ratio_threshold:
			sleep_frames += 1
			if sleep_frames >= max_sleep_frames:

				if not alarmed:
					alarmed = True

					# Phát âm thanh cảnh báo
					t = Thread(target=play_sound,
							   args=(wav_path,))
					t.deamon = True
					t.start()

				# hiện text cảnh báo trên màn hình
				cv2.putText(frame, "PHAT HIEN NGU GAT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

				#Bật LED
				GPIO.setmode(GPIO.BCM)
				GPIO.setwarnings(False)
				GPIO.setup(18,GPIO.OUT)
				for i range(3):
					GPIO.output(18,GPIO.HIGH)
					time.sleep(1)
					GPIO.output(18,GPIO.LOW)
					time.sleep(1)

		# Nếu mở mắt
		else:

			# reset parameter
			sleep_frames = 0
			alarmed = False

			# Hiện text EVR
			cv2.putText(frame, "EYE AVG RATIO: {:.3f}".format(eye_avg_ratio), (10, 30),	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

	# show lên màn hình
	cv2.imshow("Camera", frame)

	# Thoát
	key = cv2.waitKey(1)
	if key == 27 or key == ord('q') or key == ord('Q'):
		break

cv2.destroyAllWindows()
vs.stop()