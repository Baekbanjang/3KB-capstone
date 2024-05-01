from flask import Flask, render_template, Response, request, session

import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import glob
import math
import pyaudio
import wave
import threading
import datetime
import subprocess

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

pygame.init() #pygame 초기화

#전역 함수 초기화
gesture_code = None
pose_code = None
isStop = False
isRecording = None
instrument_code = '0'
gesture_preset = '1'
pose_preset = '1'
current_time=None
out = None

# 녹음 상태와 스레드를 관리하기 위한 변수
frames = []
recording_thread = None

code = {
    '0':'c_low', '1':'d', '2':'e', '3':'f', '4':'g',
    '5':'a', '6':'b', '7':'stop'
}

mode = {
    '0': 'gesture', '1': 'pose'
}

instrument = {
    '0': "piano",
    '1': "pipe"
}

sounds = {
    0: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/c_low.ogg'),
    1: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/d.ogg'),
    2: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/e.ogg'),
    3: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/f.ogg'),
    4: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/g.ogg'),
    5: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/a.ogg'),
    6: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/b.ogg')
}

# 현재 시간을 전역 변수 current_time에 업데이트하는 함수. 현재 시간은 "년-월-일_시-분-초" 형태로 형식화
def update_current_time():
    global current_time
      # 현재 시간을 글로벌 변수인 current_time에 저장. 현재 시간을 "%Y-%m-%d_%H-%M-%S"의 형식으로 포맷.
    current_time=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# 오디오 및 비디오 녹음을 시작하는 함수
def start_recording():
    global isRecording, frames, out

    # 현재 날짜 및 시간을 포맷에 맞게 저장. 해당 시간은 녹음 및 녹화된 파일의 이름에 사용
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    p = pyaudio.PyAudio()

    # 녹음 설정: 체널 수, 샘플 레이트, 버퍼 크기 등을 설정
    stream = p.open(format=pyaudio.paInt16,
                    channels=2,
                    rate=44100,
                    input=True,
                    input_device_index=1,
                    frames_per_buffer=1024)

    frames = [] # 오디오 프레임을 저장할 리스트

    # 비디오 녹화를 위한 설정. XVID 코덱을 사용
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"{output_directory}/output_{current_time}.avi", fourcc, 30.0, (640, 480))

    # 녹음이 진행되는 동안 오디오 데이터를 계속하여 frames 리스트에 추가
    while isRecording:
        data = stream.read(1024)
        frames.append(data)

    # 녹음 종료 과정
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 오디오 파일로 저장
    wf = wave.open(f"{output_directory}/output_{current_time}.wav", 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()
    frames = [] # 오디오 프레임 데이터를 초기화


# 오디오 파일과 비디오 파일을 병합하는 함수
def merge_audio_video(video_file, audio_file, output_file): # 오디오, 비디오 파일 병합 함수
    command = ['ffmpeg', '-y', '-i', video_file, '-i', audio_file, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', output_file]
    try:
        # 외부 명령어 실행. 병합 과정에서 발생하는 표준 출력과 오류는 각각 stdout, stderr에 저장.
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 병합 완료되면 원본 비디오 파일과 오디오 파일을 삭제
        os.remove(video_file)
        os.remove(audio_file)
        print(f"Deleted original files: {video_file} and {audio_file}")
    except subprocess.CalledProcessError as e:
        print("Error Occurred:", e)
        print("Error Output:", e.stderr.decode())


# 데이터셋 디렉토리 생성
data_directory = "data"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)


# 데이터셋 디렉토리 생성
instrument_directory = "instrument"
if not os.path.exists(instrument_directory):
    os.makedirs(instrument_directory)


# 저장할 디렉토리 생성
output_directory = "recordings"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# 현재 선택된 악기에 해당하는 소리 파일을 로드하여 sounds 딕셔너리를 업데이트하는 함수
def update_sounds():
    global instrument_code, sounds
    sounds = {
        0: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/c_low.ogg'),
        1: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/d.ogg'),
        2: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/e.ogg'),
        3: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/f.ogg'),
        4: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/g.ogg'),
        5: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/a.ogg'),
        6: pygame.mixer.Sound('flask_server/instrument/'+instrument[instrument_code]+'/b.ogg')
    }


# 주어진 세 랜드마크 간의 각도를 계산하여 반환, 포즈 각도를 구하는데 사용
def calculateAngle(landmark1, landmark2, landmark3):
    # 각 랜드마크의 x, y 좌표 추출
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    x3, y3 = landmark3.x, landmark3.y

    # 랜드마크2를 기준으로 랜드마크1과 랜드마크3의 각도 계산
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # 계산된 각도가 음수일 경우 360을 더하여 양수로 변환
    if angle < 0:
        angle += 360

    return angle


# 손동작 인식을 위한 데이터 수집 및 처리를 담당하는 함수
def get_gesture_set():
    # 지정된 경로의 제스처 데이터 파일이 있는지 확인하고, 파일 크기가 0보다 크면 파일을 불러옴.
    # 그렇지 않으면 빈 numpy 배열을 생성.
    global gesture_code, gesture_preset
    if os.path.exists('flask_server/data/gesture/'+ gesture_preset +'/gesture_train_'+ code[gesture_code] + '.csv') and os.path.getsize('flask_server/data/gesture/'+ gesture_preset +'/gesture_train_'+ code[gesture_code] + '.csv') > 0:
        file = np.genfromtxt('flask_server/data/gesture/'+ gesture_preset +'/gesture_train_'+ code[gesture_code] + '.csv', delimiter=',')
    else:
        file = np.empty((0, 16))

    max_num_hands = 1 # 동시에 감지할 손의 최대 개수를 설정

    # mediapipe의 제스처 모델 설정
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1) # 카메라에서 받은 이미지를 뒤집어서 사용자가 보는 것처럼 만듬.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR 이미지를 RGB로 변환.

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 이미지를 다시 BGR로 변환하여 출력 준비.

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((33, 3))  # 손의 각 점 위치를 저장할 배열.
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # 부모 점과 자식 점의 위치를 기반으로 벡터를 계산
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20,3]
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) 

                data = np.array([angle], dtype=np.float32)
                data = np.append(data, gesture_code) # 분류될 제스처 코드를 데이터에 추가.

                # 인식된 현재 포즈의 정보를 담은 데이터를 배열에 추가, 현재 촬영되는 포즈 정보를 화면에 표시.
                cv2.putText(img, f'Current Gesture: {code[gesture_code]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                file = np.vstack((file, data.astype(float)))
        if(isStop) :
            # 데이터 수집이 완료되면 현재까지 수집된 데이터를 CSV 파일로 저장.
            np.savetxt('flask_server/data/gesture/'+ gesture_preset +'/gesture_train_' + code[gesture_code] + '.csv', file, delimiter=',')
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') # 웹브라우저에 현재 프레임을 실시간으로 전송.
        

# 포즈 데이터를 수집하고, KNN 모델을 훈련시키는 함수        
def get_pose_set():
    global pose_code, pose_preset
    if os.path.exists('flask_server/data/pose/pose_angle_train_'+ code[gesture_code] + '.csv') and os.path.getsize('flask_server/data/pose/pose_angle_train_'+ code[gesture_code] + '.csv') > 0:
        file = np.genfromtxt('flask_server/data/pose/pose_angle_train_'+ code[gesture_code] + '.csv', delimiter=',')
    else:
        file = np.empty((0, 16))

    # MediaPipe pose 모델 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
    static_image_mode=False, #정적 이미지모드, 비디오 스트림 입력
    model_complexity=1, # 모델 복잡성 1
    smooth_landmarks=True, # 부드러운 랜드마크, 솔루션 필터가 지터를 줄이기 위해 다른 입력 이미지에 랜드마크 표시
    min_detection_confidence=0.5, # 최소 탐지 신뢰값, 기본 0.5
    min_tracking_confidence=0.5) # 최소 추적 신뢰값 , 기본 0.5

    # 파일에서 좌표와 레이블 데이터를 추출
    coordinate=file[:,:-1].astype(np.float32) # 각도 데이터
    label=file[:,-1].astype(np.float32) # 레이블 데이터
    knn=cv2.ml.KNearest_create()
    knn.train(coordinate, cv2.ml.ROW_SAMPLE, label) # KNN 모델 훈련

    # 카메라 캡처가 열려 있는 동안 계속 수행
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = pose.process(img)

        # 인식되는 포즈 존재 시
        if result.pose_landmarks is not None:
            # 포즈 랜드마크 생성
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # 포즈 랜드마크 그리기

            # 계산된 각도 저장
            landmarks = result.pose_landmarks.landmark
            angles = [
                # calculateAngle 함수 사용해 특정 랜드마크들을 통한 포즈 각도 계산
                calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value], landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
            ]

            # 각도 데이터와 포즈 코드 번호를 합쳐서 저장
            data = np.append(angles, code)
            file = np.vstack((file, data))
            
            # 현재 촬영되는 포즈 정보를 화면에 표시
            cv2.putText(img, f'Current Pose: {code[pose_code]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # isStop 전역 변수가 True일 경우 현재까지 수집된 데이터를 파일로 저장하고 반복을 멈추거나 추가적인 처리가 가능
        if(isStop) :
            np.savetxt('flask_server/data/pose/pose_angle_train_' + code[pose_code] + '.csv', file, delimiter=',')

        # 이미지를 JPG 형태로 인코딩하고, 바이트로 변환하여 반환    
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 제스처 인식 모델 학습 및 실시간 제스처 인식을 위한 함수
def gesture_gen():
    global gesture_preset, sounds, isRecording, out

    # 학습 데이터 파일 경로 설정
    file_path = 'flask_server/data/gesture/'+ gesture_preset +'/gesture_train.csv'

     # 기존 학습 데이터 파일이 있다면 삭제
    if os.path.exists(file_path):
        os.remove(file_path)

    #data 폴더에 있는 데이터셋들 취합
    file_list = glob.glob('flask_server/data/gesture/'+ gesture_preset +'/' + '*')
    with open('flask_server/data/gesture/'+ gesture_preset +'/gesture_train.csv', 'w') as f: #2-1.merge할 파일을 열고
        for file in file_list:
            with open(file ,'r') as f2:
                while True:
                    line = f2.readline() #2.merge 대상 파일의 row 1줄을 읽어서

                    if not line: #row가 없으면 해당 csv 파일 읽기 끝
                        break

                    f.write(line) #3.읽은 row 1줄을 merge할 파일에 쓴다.
                
            file_name = file.split('/')[-1]

    # 제스처 인식 모델 학습 파트
    if os.path.exists('flask_server/data/gesture/'+ gesture_preset +'/gesture_train.csv') and os.path.getsize('flask_server/data/gesture/'+ gesture_preset +'/gesture_train.csv') > 0:
        file = np.genfromtxt('flask_server/data/gesture/'+ gesture_preset +'/gesture_train.csv', delimiter=',')
    else:
        file = np.empty((0, 16))
    angle = file[:,:-1].astype(np.float32)
    label = file[:, -1].astype(np.float32)  
    knn = cv2.ml.KNearest_create()
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)

    max_num_hands = 1 # 최대 인식할 손의 개수

    # MediaPipe 핸드 제스처 모델 설정
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

    temp_idx = None  # 직전에 인식한 제스처 인덱스 저장 변수

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))  # 손가락 조인트 포인트 초기화
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # 조인트 간의 각도 계산
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20,3]

                # 각도 벡터를 이용한 학습 데이터 생성 및 제스처 인식 모델 적용
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # 각도 변환

                # 인식된 제스처에 따른 소리 재생 처리
                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])
                if temp_idx != idx :
                    temp_idx = idx
                    pygame.mixer.stop()
                    if idx in sounds:
                        sound = sounds[idx]
                        sound.set_volume(0.3)
                        sound.play()
                    elif idx == 13:
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.stop()

                mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS) # 인식된 손에 랜드마크 그리기

        if((isRecording == True) and (isRecording != None)) :
            out.write(img) # 비디오 녹화 상태일 경우, 현재 이미지를 비디오에 기록
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes() # 프레임을 바이트 형태로 인코딩하여 스트리밍을 위한 준비
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 스트리밍을 위한 프레임 데이터 생성


# 포즈 인식 모델 학습 및 실시간 포즈 인식을 위한 함수
def pose_gen():
    global pose_preset, sounds
    # 기존에 수집된 데이터셋 초기화
    file_path = 'flask_server/data/pose/'+ pose_preset +'/pose_angle_train.csv'
    if os.path.exists(file_path):
        os.remove(file_path)

    # data 폴더에 있는 데이터셋들 취합
    file_list = glob.glob('flask_server/data/pose/'+ pose_preset +'/' + '*')
    with open('flask_server/data/pose/'+ pose_preset +'/pose_angle_train.csv', 'w') as f: # 취합할 파일을 열고
        for file in file_list:
            with open(file ,'r') as f2:
                while True:
                    line = f2.readline() # 대상 파일의 row를 1줄 읽고

                    if not line: # row가 없으면 해당 csv 파일 읽기 끝
                        break

                    f.write(line) # 읽은 row를 취합할 파일에 쓴다.
                
            file_name = file.split('\\')[-1]

    # 포즈 인식 모델 로드
    if os.path.exists('flask_server/data/pose/'+ pose_preset +'/pose_angle_train.csv') and os.path.getsize('flask_server/data/pose/'+ pose_preset +'/pose_angle_train.csv') > 0:
        file = np.genfromtxt('flask_server/data/pose/'+ pose_preset +'/pose_angle_train.csv', delimiter=',')
    else:
        file = np.empty((0, 9))

    coordinate=file[:,:-1].astype(np.float32) # 각도 데이터
    label=file[:,-1].astype(np.float32) # 레이블 데이터
    knn=cv2.ml.KNearest_create()
    knn.train(coordinate, cv2.ml.ROW_SAMPLE, label) # KNN 모델 훈련

    # MediaPipe pose 모델 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
    static_image_mode=False, #정적 이미지모드, 비디오 스트림 입력
    model_complexity=1, # 모델 복잡성 1
    smooth_landmarks=True, # 부드러운 랜드마크, 솔루션 필터가 지터를 줄이기 위해 다른 입력 이미지에 랜드마크 표시
    min_detection_confidence=0.5, # 최소 탐지 신뢰값, 기본 0.5
    min_tracking_confidence=0.5) # 최소 추적 신뢰값 , 기본 0.5

    temp_idx = None
    while cap.isOpened():
        ret, img=cap.read()
        if not ret:
            continue

        img=cv2.flip(img, 1) # 이미지를 좌우 반전
        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR 이미지를 RGB로 변환

        result=pose.process(imgRGB)

        if result.pose_landmarks is not None:
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # 포즈 랜드마크 그리기

            # 필요한 랜드마크 간의 각도 계산
            landmarks = result.pose_landmarks.landmark
            angle1 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
            angle2 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
            angle3 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value])
            angle4 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value], landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value])
            angle5 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
            angle6 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
            angle7 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
            angle8 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])

            # 각도 데이터를 배열에 저장
            pose_array = np.array([angle1, angle2, angle3, angle4, angle5, angle6, angle7, angle8])  # 각도 계산 결과를 배열에 추가. 필요한 각도 수에 맞게 조정하세요.
            pose_array = pose_array.reshape((1, -1)).astype(np.float32)  # KNN 모델에 입력하기 위한 형태로 변환

            # 포즈 인식 및 해당 포즈에 맞는 음악 재생
            ret, results, neighbours, dist = knn.findNearest(pose_array, 3)  # KNN을 사용하여 가장 가까운 포즈 인식
            idx = int(results[0][0])
            if temp_idx != idx :
                temp_idx = idx
                pygame.mixer.stop()
                if idx in sounds:
                    sound = sounds[idx]
                    sound.set_volume(0.3)
                    sound.play()
                elif idx == 13:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/Get_BodyMovements', methods=['GET', 'POST'])
def process_body_data():
    '''
    포즈 데이터 처리를 위한 엔드포인트.
    클라이언트로부터 전송받은 데이터에 따라 신체 움직임 데이터의 레코딩 시작, 정지, 삭제 및 프리셋 변경을 관리
    '''
    global pose_code, isStop, pose_preset
    if request.method == 'POST':
        if 'preset' in request.form:
            pose_preset = request.form['preset'] # 프리셋 변경 요청 처리
            return render_template('GetHandDataSet.html', message="프리셋 변경")
        
        if 'button_value' in request.form: # 데이터 레코딩 시작 요청 처리
            pose_code = request.form['button_value'] 
            session['button_value_received'] = True  # 세션에 상태 저장하여 데이터 레코딩 상태를 관리
            return render_template('GetBodyDataSet.html', message="웹캠 작동을 시작합니다.")
        
        if 'stop_sign' in request.form and session.get('button_value_received'): # 데이터 레코딩 정지 요청 처리
            isStop = request.form['stop_sign']
            session['button_value_received'] = False # 세션 상태 초기화
            return render_template('GetBodyDataSet.html', message="웹캠 작동을 종료합니다.")
        
        if 'delete_button_value' in request.form: # 특정 데이터 삭제 요청 처리
            pose_code = request.form['delete_button_value']
            file_path = 'data/pose/'+ pose_preset +'/pose_angle_train_'+ code[pose_code] + '.csv'
            if os.path.exists(file_path):
                os.remove(file_path) # 파일 존재 시 삭제
            return render_template('GetBodyDataSet.html', message="파일을 삭제합니다.")
        
    return render_template('GetBodyDataSet.html') # POST 요청이 아닌 경우 기본 페이지 로드


@app.route('/Get_HandGestures', methods=['GET', 'POST'])
def process_gesture_data():
    '''
    손동작 데이터 처리를 위한 엔드포인트.
    클라이언트로부터 전송받은 데이터에 따라 손동작 데이터의 레코딩 시작, 정지, 삭제 및 프리셋 변경을 관리한다.
    '''
    global gesture_code, isStop, gesture_preset, isRecording, out, recording_thread, current_time
    if request.method == 'POST':
        if 'preset' in request.form: # 프리셋 변경 요청 처리
            gesture_preset = request.form['preset']
            return render_template('GetHandDataSet.html', message="프리셋 변경")
        
        if 'button_value' in request.form:  # 데이터 레코딩 시작 요청 처리
            gesture_code = request.form['button_value']
            session['button_value_received'] = True  # 세션에 상태 저장
            return render_template('GetHandDataSet.html', message="웹캠 작동을 시작합니다.")
        
        if 'stop_sign' in request.form and session.get('button_value_received'): # 데이터 레코딩 정지 요청 처리
            isStop = request.form['stop_sign']
            session['button_value_received'] = False  # 세션 상태 초기화
            return render_template('GetHandDataSet.html', message="웹캠 작동을 종료합니다.")
        
        if 'delete_button_value' in request.form: # 특정 데이터 삭제 요청 처리
            gesture_code = request.form['delete_button_value']
            file_path = 'data/gesture/'+ gesture_preset +'/gesture_train_'+ code[gesture_code] + '.csv'
            if os.path.exists(file_path):
                os.remove(file_path) # 파일 존재 시 삭제
            return render_template('GetHandDataSet.html', message="파일을 삭제합니다.")
        
    return render_template('GetHandDataSet.html')  # POST 요청이 아닌 경우 기본 페이지 로드


@app.route('/HandGestures_play', methods=['GET', 'POST'])
def hand_gestures_play():
    # 사용자가 손동작을 이용해 음악을 연주하고 녹화할 수 있게 하는 엔드포인트
    global instrument_code, gesture_preset, isRecording, out, recording_thread, current_time
    if request.method == 'POST':
        if 'preset' in request.form:
            gesture_preset = request.form['preset']
            return render_template('HandPlay.html', message="프리셋 변경")
        
        if 'instrument_value' in request.form:
            instrument_code = request.form['instrument_value']
            update_sounds()
            return render_template('HandPlay.html', message="악기 변경")
        
        if 'isRecording' in request.form:
            if(request.form['isRecording'] == 'True') :
                update_current_time()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(f"{output_directory}/output_{current_time}.avi", fourcc, 30.0, (640, 480))
                isRecording = True
                recording_thread = threading.Thread(target=start_recording)
                recording_thread.start()
                return render_template('HandPlay.html', message="녹화를 시작합니다.")
            elif(request.form['isRecording'] == 'False') :
                isRecording = None
                out.release()
                out = None
                recording_thread.join()
                merge_audio_video(f"{output_directory}/output_{current_time}.avi", f"{output_directory}/output_{current_time}.wav", f"{output_directory}/final_output_{current_time}.mp4")
                return render_template('HandPlay.html', message="녹화를 종료합니다.")
            
    return render_template('HandPlay.html')

@app.route('/BodyMovements_play')   
def body_movements_play():
    # 사용자가 신체 움직임을 이용해 음악을 연주할 수 있는 엔드포인트
    global instrument_code, pose_preset
    if request.method == 'POST':
        if 'preset' in request.form:
            pose_preset = request.form['preset']
            return render_template('BodyPlay.html', message="프리셋 변경")
        
        if 'instrument_value' in request.form:
            instrument_code = request.form['instrument_value']
            update_sounds()
            return render_template('BodyPlay.html', message="악기 변경")
        
    return render_template('BodyPlay.html')


@app.route('/processed_video_gesture')
def processed_video_gesture():
    # 엔드포인트를 통해 제스처 처리된 비디오 스트리밍을 제공
    # gesture_gen() 함수에서 생성된 스트리밍 데이터를 클라이언트에 전송
    # MIME 타입은 'multipart/x-mixed-replace'를 사용하여 지속적인 데이터 스트림을 구현
    return Response(gesture_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_video_pose')
def processed_video_pose():
    # 엔드포인트를 통해 포즈 처리된 비디오 스트리밍을 제공
    # pose_gen() 함수에서 생성된 스트리밍 데이터를 클라이언트에 전송
    # 해당 엔드포인트도 지속적인 비디오 스트림을 위해 같은 MIME 타입을 사용
    return Response(pose_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_video_gesture')
def get_video_gesture():
    # 해당 엔드포인트에서는 제스처 코드에 따라 특정 제스처 비디오 세트를 스트리밍.
    global gesture_code # 제스처 코드를 글로벌 변수로 사용.
    # get_gesture_set() 함수를 통해 제스처 비디오 세트를 스트리밍.
    return Response(get_gesture_set(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_video_pose')
def get_video_pose():
    # 해당 엔드포인트에서는 포즈 데이터에 맞는 비디오 세트를 스트리밍하는 기능을 제공.
    # get_pose_set() 함수를 통해 포즈 비디오 세트를 스트리밍.
    return Response(get_pose_set(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True) # 애플리케이션을 디버그 모드로 실행