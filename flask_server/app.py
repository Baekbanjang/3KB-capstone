from flask import Flask, render_template, Response, request, session

import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import glob
import math

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

pygame.init()
gesture_code = None
pose_code = None
isStop = False
instrument_code = '0'

code = {
    '0':'c4', '1':'c4_shop', '2':'d4', '3':'d4_shop', '4':'e4',
    '5':'f4', '6':'f4_shop', '7':'g4', '8':'g4_shop', '9':'a4',
    '10':'a4_shop', '11':'b4', '12':'c5', '13':'stop'
}

mode = {
    '0': 'gesture', '1': 'pose'
}

pose_sounds = {
    0: pygame.mixer.Sound('instrument/piano/c4.ogg'),
    1: pygame.mixer.Sound('instrument/piano/d4.ogg'),
    2: pygame.mixer.Sound('instrument/piano/e4.ogg'),
    3: pygame.mixer.Sound('instrument/piano/f4.ogg'),
    4: pygame.mixer.Sound('instrument/piano/g4.ogg'),
    5: pygame.mixer.Sound('instrument/piano/a4.ogg'),
    6: pygame.mixer.Sound('instrument/piano/b4.ogg'),
}

instrument = {
    '0': "piano",
    '1': "pipe"
}

pipe = {
    0: pygame.mixer.Sound('instrument/pipe/c2.wav'),
    2: pygame.mixer.Sound('instrument/pipe/d2.wav'),
    4: pygame.mixer.Sound('instrument/pipe/e2.wav'),
    5: pygame.mixer.Sound('instrument/pipe/f2.wav'),
    7: pygame.mixer.Sound('instrument/pipe/g2.wav'),
    9: pygame.mixer.Sound('instrument/pipe/a2.wav'),
    11: pygame.mixer.Sound('instrument/pipe/b2.wav'),
    12: pygame.mixer.Sound('instrument/pipe/c3.wav')
}

piano = {
    0: pygame.mixer.Sound('instrument/piano/c4.ogg'),
    1: pygame.mixer.Sound('instrument/piano/c4_shop.ogg'),
    2: pygame.mixer.Sound('instrument/piano/d4.ogg'),
    3: pygame.mixer.Sound('instrument/piano/d4_shop.ogg'),
    4: pygame.mixer.Sound('instrument/piano/e4.ogg'),
    5: pygame.mixer.Sound('instrument/piano/f4.ogg'),
    6: pygame.mixer.Sound('instrument/piano/f4_shop.ogg'),
    7: pygame.mixer.Sound('instrument/piano/g4.ogg'),
    8: pygame.mixer.Sound('instrument/piano/g4_shop.ogg'),
    9: pygame.mixer.Sound('instrument/piano/a4.ogg'),
    10: pygame.mixer.Sound('instrument/piano/a4_shop.ogg'),
    11: pygame.mixer.Sound('instrument/piano/b4.ogg'),
    12: pygame.mixer.Sound('instrument/piano/c5.ogg')
}

def calculateAngle(landmark1, landmark2, landmark3):
    # Get the required landmarks coordinates.
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    x3, y3 = landmark3.x, landmark3.y

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle

def get_gesture_set():
    global gesture_code
    if os.path.exists('data/gesture/gesture_train_'+ code[gesture_code] + '.csv') and os.path.getsize('data/gesture/gesture_train_'+ code[gesture_code] + '.csv') > 0:
        file = np.genfromtxt('data/gesture/gesture_train_'+ code[gesture_code] + '.csv', delimiter=',')
    else:
        file = np.empty((0, 16))

    max_num_hands = 1

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)

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
                joint = np.zeros((33, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                data = np.array([angle], dtype=np.float32)
                data = np.append(data, gesture_code) #인덱스에 코드 번호 추가

                # 현재 촬영되는 포즈 정보를 화면에 표시
                cv2.putText(img, f'Current Gesture: {code[gesture_code]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                file = np.vstack((file, data.astype(float)))
        if(isStop) :
            np.savetxt('data/gesture/gesture_train_' + code[gesture_code] + '.csv', file, delimiter=',')
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
def get_pose_set():
    global pose_code
    if os.path.exists('data/pose/pose_angle_train_'+ code[gesture_code] + '.csv') and os.path.getsize('data/pose/pose_angle_train_'+ code[gesture_code] + '.csv') > 0:
        file = np.genfromtxt('data/pose/pose_angle_train_'+ code[gesture_code] + '.csv', delimiter=',')
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

    coordinate=file[:,:-1].astype(np.float32) # 각도 데이터
    label=file[:,-1].astype(np.float32) # 레이블 데이터
    knn=cv2.ml.KNearest_create()
    knn.train(coordinate, cv2.ml.ROW_SAMPLE, label) # KNN 모델 훈련

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = pose.process(img)
        if result.pose_landmarks is not None:
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # 포즈 랜드마크 그리기

            landmarks = result.pose_landmarks.landmark
            angles = [
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
        if(isStop) :
            np.savetxt('data/pose/pose_angle_train_' + code[pose_code] + '.csv', file, delimiter=',')
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gesture_gen():
    global instrument_code
    file_path = 'data/gesture/gesture_train.csv'
    if os.path.exists(file_path):
        os.remove(file_path)

    #data 폴더에 있는 데이터셋들 취합
    file_list = glob.glob('data/gesture/' + '*')
    with open('data/gesture/gesture_train.csv', 'w') as f: #2-1.merge할 파일을 열고
        for file in file_list:
            with open(file ,'r') as f2:
                while True:
                    line = f2.readline() #2.merge 대상 파일의 row 1줄을 읽어서

                    if not line: #row가 없으면 해당 csv 파일 읽기 끝
                        break

                    f.write(line) #3.읽은 row 1줄을 merge할 파일에 쓴다.
                
            file_name = file.split('/')[-1]

    # Gesture recognition model
    if os.path.exists('data/gesture/gesture_train.csv') and os.path.getsize('data/gesture/gesture_train.csv') > 0:
        file = np.genfromtxt('data/gesture/gesture_train.csv', delimiter=',')
    else:
        file = np.empty((0, 16))
    angle = file[:,:-1].astype(np.float32)
    label = file[:, -1].astype(np.float32)  
    knn = cv2.ml.KNearest_create()
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)

    max_num_hands = 1

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

    temp_idx = None
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)

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
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])
                selected_instrument_name = instrument[instrument_code]
                if selected_instrument_name == "piano":
                    selected_instrument = piano
                elif selected_instrument_name == "pipe":
                    selected_instrument = pipe
                if temp_idx != idx :
                    temp_idx = idx
                    pygame.mixer.stop()
                    if idx in selected_instrument:
                        sound = selected_instrument[idx]
                        sound.set_volume(0.3)
                        sound.play(-1)
                    elif idx == 13:
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.stop()
                mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def pose_gen():
    # 기존에 수집된 데이터셋 초기화
    file_path = 'data/pose/pose_angle_train.csv'
    if os.path.exists(file_path):
        os.remove(file_path)

    # data 폴더에 있는 데이터셋들 취합
    file_list = glob.glob('data/pose/' + '*')
    with open('data/pose/pose_angle_train.csv', 'w') as f: # 취합할 파일을 열고
        for file in file_list:
            with open(file ,'r') as f2:
                while True:
                    line = f2.readline() # 대상 파일의 row를 1줄 읽고

                    if not line: # row가 없으면 해당 csv 파일 읽기 끝
                        break

                    f.write(line) # 읽은 row를 취합할 파일에 쓴다.
                
            file_name = file.split('\\')[-1]

    # 포즈 인식 모델 로드
    if os.path.exists('data/pose/pose_angle_train.csv') and os.path.getsize('data/pose/pose_angle_train.csv') > 0:
        file = np.genfromtxt('data/pose/pose_angle_train.csv', delimiter=',')
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
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)

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
                if idx in pose_sounds:
                    sound = pose_sounds[idx]
                    sound.set_volume(0.3)
                    sound.play(-1)
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
    global pose_code, isStop
    if request.method == 'POST':
        if 'button_value' in request.form:
            pose_code = request.form['button_value']
            session['button_value_received'] = True  # 세션에 상태 저장
            return render_template('GetBodyDataSet.html', message="녹화를 시작합니다.")
        if 'stop_sign' in request.form and session.get('button_value_received'):
            isStop = request.form['stop_sign']
            session['button_value_received'] = False  # 상태 초기화
            return render_template('GetBodyDataSet.html', message="녹화를 종료합니다.")
        if 'delete_button_value' in request.form:
            pose_code = request.form['delete_button_value']
            file_path = 'data/pose/pose_angle_train_'+ code[pose_code] + '.csv'
            if os.path.exists(file_path):
                os.remove(file_path)
            return render_template('GetBodyDataSet.html', message="파일을 삭제합니다.")
    return render_template('GetBodyDataSet.html')

@app.route('/Get_HandGestures', methods=['GET', 'POST'])
def process_gesture_data():
    global gesture_code, isStop
    if request.method == 'POST':
        if 'button_value' in request.form:
            gesture_code = request.form['button_value']
            session['button_value_received'] = True  # 세션에 상태 저장
            return render_template('GetHandDataSet.html', message="녹화를 시작합니다.")
        if 'stop_sign' in request.form and session.get('button_value_received'):
            isStop = request.form['stop_sign']
            session['button_value_received'] = False  # 상태 초기화
            return render_template('GetHandDataSet.html', message="녹화를 종료합니다.")
        if 'delete_button_value' in request.form:
            gesture_code = request.form['delete_button_value']
            file_path = 'data/gesture/gesture_train_'+ code[gesture_code] + '.csv'
            if os.path.exists(file_path):
                os.remove(file_path)
            return render_template('GetHandDataSet.html', message="파일을 삭제합니다.")
    return render_template('GetHandDataSet.html')

@app.route('/HandGestures_play', methods=['GET', 'POST'])
def hand_gestures_play():
    global instrument_code
    if request.method == 'POST':
        if 'instrument_value' in request.form:
            instrument_code = request.form['instrument_value']
            return render_template('HandPlay.html', message="악기 변경")
    return render_template('HandPlay.html')

@app.route('/BodyMovements_play')   
def body_movements_play():
    return render_template('BodyPlay.html')

@app.route('/processed_video_gesture')
def processed_video_gesture():
    return Response(gesture_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_video_pose')
def processed_video_pose():
    return Response(pose_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_video_gesture')
def get_video_gesture():
    global gesture_code
    return Response(get_gesture_set(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_video_pose')
def get_video_pose():
    return Response(get_pose_set(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)