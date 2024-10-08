import cv2, time, pygame, mediapipe as mp, numpy as np, os, glob, math, threading, cv2
from utils.global_vars import get_global_var, cap, cap2, code, set_global_var, gesture_sounds1, gesture_sounds2, pose_sounds
from utils.audio_utils import update_gesture_sounds1, update_gesture_sounds2, update_pose_sounds

# 파일 접근 동기화를 위한 Lock 객체 생성
file_lock = threading.Lock()

def load_and_train_knn(preset):
    file_path = 'flask_server/data/gesture/' + preset + '/gesture_train.csv'
    
    with file_lock:
        if os.path.exists(file_path):
            os.remove(file_path)

        # Collect datasets in the data folder
        file_list = glob.glob('flask_server/data/gesture/' + preset + '/' + '*')
        with open(file_path, 'w') as f:  # Open file to merge data
            for file in file_list:
                with open(file, 'r') as f2:
                    while True:
                        line = f2.readline()  # Read a row from the merge target file
                        if not line:  # End of the csv file
                            break
                        f.write(line)  # Write the row to the merge file

    # Gesture recognition model
    with file_lock:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            file = np.genfromtxt(file_path, delimiter=',')
        else:
            file = np.empty((0, 16))
            return None

    angle = file[:, :-1].astype(np.float32)
    label = file[:, -1].astype(np.float32)

    knn = cv2.ml.KNearest_create()
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)

    return knn

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
    gesture_code = get_global_var('gesture_code')
    gesture_preset = get_global_var('gesture_preset')
    if os.path.exists('flask_server/data/gesture/'+ gesture_preset +'/gesture_train_'+ code[gesture_code] + '.csv') and os.path.getsize('flask_server/data/gesture/'+ gesture_preset +'/gesture_train_'+ code[gesture_code] + '.csv') > 0:
        file = np.genfromtxt('flask_server/data/gesture/'+ gesture_preset +'/gesture_train_'+ code[gesture_code] + '.csv', delimiter=',')
    else:
        file = np.empty((0, 16))

    max_num_hands = 1

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

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
                mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)
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

                cv2.imwrite(f'flask_server/static/capture/gesture/'+ gesture_preset +'/capture_gesture_'+code[gesture_code]+'.jpg', img)

                # 현재 촬영되는 포즈 정보를 화면에 표시
                cv2.putText(img, f'Current Gesture: {code[gesture_code]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                file = np.vstack((file, data.astype(float)))
        if(get_global_var('isStop')) :
            np.savetxt('flask_server/data/gesture/'+ gesture_preset +'/gesture_train_' + code[gesture_code] + '.csv', file, delimiter=',')
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
def get_pose_set():
    pose_preset = get_global_var('pose_preset')
    pose_code = get_global_var('pose_code')
    if os.path.exists('flask_server/data/pose/'+ pose_preset +'/pose_angle_train_'+ code[pose_code] + '.csv') and os.path.getsize('flask_server/data/pose/'+ pose_preset +'/pose_angle_train_'+ code[pose_code] + '.csv') > 0:
        file = np.genfromtxt('flask_server/data/pose/'+ pose_preset +'/pose_angle_train_'+ code[pose_code] + '.csv', delimiter=',')
    else:
        file = np.empty((0,17))
    # MediaPipe pose 모델 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False, #정적 이미지모드, 비디오 스트림 입력
        model_complexity=1, # 모델 복잡성 1
        smooth_landmarks=True, # 부드러운 랜드마크, 솔루션 필터가 지터를 줄이기 위해 다른 입력 이미지에 랜드마크 표시
        min_detection_confidence=0.5, # 최소 탐지 신뢰값, 기본 0.5
        min_tracking_confidence=0.5) # 최소 추적 신뢰값 , 기본 0.5

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = pose.process(img)
        if result.pose_landmarks is not None:
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # 포즈 랜드마크 그리기

            landmarks = result.pose_landmarks.landmark
            angles = [
                calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value], landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]),
                calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
            ]

            # 각도 데이터와 포즈 코드 번호를 합쳐서 저장
            data = np.append(angles, pose_code)
            file = np.vstack((file, data.astype(float)))

            cv2.imwrite(f'flask_server/static/capture/pose/'+ pose_preset +'/capture_pose_'+code[pose_code]+'.jpg', img)
            
            # 현재 촬영되는 포즈 정보를 화면에 표시
            cv2.putText(img, f'Current Pose: {code[pose_code]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if(get_global_var('isStop')) :
            np.savetxt('flask_server/data/pose/'+ pose_preset +'/pose_angle_train_' + code[pose_code] + '.csv', file, delimiter=',')
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gesture_gen():
    gesture_left_octave_code = get_global_var('gesture_left_octave_code')
    current_gesture_preset = get_global_var('gesture_preset')
    knn = load_and_train_knn(current_gesture_preset)

    max_num_hands = 1

    frame_count = 0
    start_time = time.time()  # 시작 시간 기록

    max_octave = get_global_var('max_octave')
    min_octave = get_global_var('min_octave')

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    temp_idx = None
    current_channel = None

    while cap.isOpened():
        if get_global_var('gesture_preset') != current_gesture_preset:
            current_gesture_preset = get_global_var('gesture_preset')
            knn = load_and_train_knn(current_gesture_preset)
        ret, img = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        current_time = time.time() - start_time  # 현재 경과된 시간 계산
        fps = round(frame_count / current_time)  # 현재 주사율 계산

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)
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
                if knn is not None :
                    ret, results, neighbours, dist = knn.findNearest(data, 3)
                    idx = int(results[0][0])
                    if temp_idx != idx :
                        if current_channel:
                            current_channel.stop()
                        temp_idx = idx
                        channel_number = idx % 8  # 0에서 7까지의 채널 사용
                        current_channel = pygame.mixer.Channel(channel_number)

                        if current_channel is None:
                            current_channel = pygame.mixer.Channel(idx)

                        current_channel.stop()
                        
                        if idx in gesture_sounds1:
                            sound = gesture_sounds1[idx]
                            if sound is not None:
                                sound.set_volume(0.3)
                                current_channel.play(sound)
                            else:
                                print(f"Sound for gesture {idx} is not available.")
                        elif idx == 8:
                            gesture_left_octave_code += 1
                            if gesture_left_octave_code > max_octave:  # 여기서 2는 octave_code의 최대값
                                gesture_left_octave_code = max_octave  # 최대값 초과 시 octave_code를 최대값으로 설정
                            gesture_left_octave_code_str = str(gesture_left_octave_code)
                            update_gesture_sounds1(gesture_left_octave_code_str)
                        elif idx == 9:
                            gesture_left_octave_code -= 1
                            if gesture_left_octave_code < min_octave:  # 여기서 2는 octave_code의 최대값
                                gesture_left_octave_code = min_octave  # 최대값 초과 시 octave_code를 최대값으로 설정
                            set_global_var('gesture_left_octave_code', gesture_left_octave_code)
                            gesture_left_octave_code_str = str(gesture_left_octave_code)
                            update_gesture_sounds1(gesture_left_octave_code_str)
                        elif idx == 13:
                            pygame.mixer.music.stop()
                        temp_idx = idx
        if get_global_var('isRecording'):
            out = get_global_var('out')
            out.write(img)
        # 프레임에 주사율 표시
        cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
# 웹캠
def gesture_gen_2():
    gesture_right_octave_code = get_global_var('gesture_right_octave_code')
    current_gesture_preset = get_global_var('gesture_preset')
    knn = load_and_train_knn(current_gesture_preset)

    max_num_hands = 1

    frame_count = 0
    start_time = time.time()  # 시작 시간 기록

    max_octave = get_global_var('max_octave')
    min_octave = get_global_var('min_octave')

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    temp_idx = None
    current_channel = None

    while cap2.isOpened():
        if get_global_var('gesture_preset') != current_gesture_preset:
            current_gesture_preset = get_global_var('gesture_preset')
            knn = load_and_train_knn(current_gesture_preset)
        ret2, img2 = cap2.read()
        if not ret2:
            continue

        frame_count += 1
        current_time = time.time() - start_time  # 현재 경과된 시간 계산
        fps = round(frame_count / current_time)  # 현재 주사율 계산

        img2 = cv2.flip(img2, 1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        result = hands.process(img2)

        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img2, res, mp_hands.HAND_CONNECTIONS)
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0,
                            13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                            13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10,
                                                12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)
                if knn is not None :
                    ret2, results, neighbours, dist = knn.findNearest(data, 3)
                    idx = int(results[0][0])

                    if temp_idx != idx:
                        if current_channel:
                            current_channel.stop()

                        channel_number = 8 + (idx % 8)  # 8에서 15까지의 채널 사용
                        current_channel = pygame.mixer.Channel(channel_number)

                        if current_channel is None:
                            current_channel = pygame.mixer.Channel(idx + 8)

                        current_channel.stop()

                        if idx in gesture_sounds2:
                            sound = gesture_sounds2[idx]
                            if sound is not None:
                                sound.set_volume(0.3)
                                current_channel.play(sound)
                            else:
                                print(f"Sound for gesture {idx} is not available.")

                        elif idx == 8:
                            gesture_right_octave_code += 1
                            if gesture_right_octave_code > max_octave:  # 여기서 2는 octave_code의 최대값
                                gesture_right_octave_code = max_octave  # 최대값 초과 시 octave_code를 최대값으로 설정
                            set_global_var('gesture_right_octave_code', gesture_right_octave_code)
                            gesture_right_octave_code_str = str(gesture_right_octave_code)
                            update_gesture_sounds2(gesture_right_octave_code_str)
                        elif idx == 9:
                            gesture_right_octave_code -= 1
                            if gesture_right_octave_code < min_octave:  # 여기서 2는 octave_code의 최대값
                                gesture_right_octave_code = min_octave  # 최대값 초과 시 octave_code를 최대값으로 설정
                            set_global_var('gesture_right_octave_code', gesture_right_octave_code)
                            gesture_right_octave_code_str = str(gesture_right_octave_code)
                            update_gesture_sounds2(gesture_right_octave_code_str)
                        elif idx == 13:
                            pygame.mixer.music.stop()
                        temp_idx = idx
        if get_global_var('isRecording'):
            out2 = get_global_var('out2')
            out2.write(img2)

        # 프레임에 주사율 표시
        cv2.putText(img2, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret2, jpeg = cv2.imencode('.jpg', img2)
        frame2 = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

def pose_gen():
    pose_octave_code = get_global_var('pose_octave_code')
    max_octave = get_global_var('max_octave')
    min_octave = get_global_var('min_octave')

    def pose_load_and_train_knn(preset):   
        # 기존에 수집된 데이터셋 초기화
        file_path = 'flask_server/data/pose/'+ preset +'/pose_angle_train.csv'
        if os.path.exists(file_path):
            os.remove(file_path)
       
        # data 폴더에 있는 데이터셋들 취합
        file_list = glob.glob('flask_server/data/pose/'+ preset +'/' + '*')
        with open('flask_server/data/pose/'+ preset +'/pose_angle_train.csv', 'w') as f: # 취합할 파일을 열고
            for file in file_list:
                with open(file ,'r') as f2:
                    while True:
                        line = f2.readline() # 대상 파일의 row를 1줄 읽고

                        if not line: # row가 없으면 해당 csv 파일 읽기 끝
                            break

                        f.write(line) # 읽은 row를 취합할 파일에 쓴다.
                
                file_name = file.split('\\')[-1]

        # 포즈 인식 모델 로드
        if os.path.exists('flask_server/data/pose/'+ preset +'/pose_angle_train.csv') and os.path.getsize('flask_server/data/pose/'+ preset +'/pose_angle_train.csv') > 0:
            file = np.genfromtxt('flask_server/data/pose/'+ preset +'/pose_angle_train.csv', delimiter=',')
        else:
            file = np.empty((0, 17))
            return None

        coordinate=file[:,:-1].astype(np.float32) # 각도 데이터
        label=file[:,-1].astype(np.float32) # 레이블 데이터
        knn=cv2.ml.KNearest_create()
        knn.train(coordinate, cv2.ml.ROW_SAMPLE, label) # KNN 모델 훈련

        return knn

    current_pose_preset = get_global_var('pose_preset')
    knn = pose_load_and_train_knn(current_pose_preset)

    # MediaPipe pose 모델 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False, #정적 이미지모드, 비디오 스트림 입력
        model_complexity=1, # 모델 복잡성 1
        smooth_landmarks=True, # 부드러운 랜드마크, 솔루션 필터가 지터를 줄이기 위해 다른 입력 이미지에 랜드마크 표시
        min_detection_confidence=0.5, # 최소 탐지 신뢰값, 기본 0.5
        min_tracking_confidence=0.5) # 최소 추적 신뢰값 , 기본 0.5

    frame_count = 0
    start_time = time.time()  # 시작 시간 기록

    temp_idx = None
    while cap.isOpened():
        if get_global_var('pose_preset') != current_pose_preset:
            current_pose_preset = get_global_var('pose_preset')
            knn = pose_load_and_train_knn(current_pose_preset)
        ret, img=cap.read()
        if not ret:
            continue
        frame_count += 1
        current_time = time.time() - start_time  # 현재 경과된 시간 계산
        fps = round(frame_count / current_time)  # 현재 주사율 계산

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
            angle5 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
            angle6 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
            angle7 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
            angle8 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            angle9 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
            angle10 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
            angle11 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
            angle12 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
            angle13 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
            angle14 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
            angle15 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
            angle16 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])

            # 각도 데이터를 배열에 저장
            pose_array = np.array([angle1, angle2, angle3, angle4, angle5, angle6, angle7, angle8, angle9, angle10, angle11, angle12, angle13, angle14, angle15, angle16])  # 각도 계산 결과를 배열에 추가. 필요한 각도 수에 맞게 조정하세요.
            pose_array = pose_array.reshape((1, -1)).astype(np.float32)  # KNN 모델에 입력하기 위한 형태로 변환

            if knn is not None :
                # 포즈 인식 및 해당 포즈에 맞는 음악 재생
                ret, results, neighbours, dist = knn.findNearest(pose_array, 3)  # KNN을 사용하여 가장 가까운 포즈 인식
                idx = int(results[0][0])
                if temp_idx != idx :
                    temp_idx = idx
                    pygame.mixer.stop()
                    if idx in pose_sounds:
                        sound = pose_sounds[idx]
                        if sound is not None:
                            sound.set_volume(0.3)
                            sound.play()
                    elif idx == 8:
                        pose_octave_code += 1
                        if pose_octave_code > max_octave:  # 여기서 2는 octave_code의 최대값
                            pose_octave_code = max_octave  # 최대값 초과 시 octave_code를 최대값으로 설정
                        pose_octave_code_str = str(pose_octave_code)
                        set_global_var('pose_octave_code', pose_octave_code)
                        update_pose_sounds(pose_octave_code_str)
                    elif idx == 9:
                        pose_octave_code -= 1
                        if pose_octave_code < min_octave:  # 여기서 2는 octave_code의 최대값
                            pose_octave_code = min_octave  # 최대값 초과 시 octave_code를 최대값으로 설정
                        pose_octave_code_str = str(pose_octave_code)
                        set_global_var('pose_octave_code', pose_octave_code)
                        update_pose_sounds(pose_octave_code_str)
                    elif idx == 13:
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.stop()
            if get_global_var('isRecording'):
                out = get_global_var('out')
                out.write(img)
            # 프레임에 주사율 표시
            cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')