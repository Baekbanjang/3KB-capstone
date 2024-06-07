from flask import Flask, render_template, Response, request, session, jsonify, send_from_directory, redirect, url_for
from pymongo import MongoClient
from bson.objectid import ObjectId
from moviepy.editor import VideoFileClip #재생 시간 확인 용도
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"]="0"
import glob
import math
import pyaudio
import wave
import threading
import datetime
import subprocess
import time
import pymongo
import bson
import csv
from gridfs import GridFS

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
client = MongoClient("mongodb://localhost:27017/")
db = client["record_videos"]
fs = GridFS(db)
fs_files_collection = db['fs.files']  # GridFS의 메타데이터 컬렉션

# 비디오 녹화를 위한 설정.
fourcc = cv2.VideoWriter_fourcc(*'XVID')

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap2 = cv2.VideoCapture(1, cv2.CAP_MSMF)

width, height = (640, 480)
set_fps = 30

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width), cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height), cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, set_fps), cap2.set(cv2.CAP_PROP_FPS, set_fps)

sort_by = 'creationDate'
sort_direction = 'asc'

pygame.init()
pygame.mixer.init()
pygame.mixer.set_num_channels(16)
gesture_code = None
pose_code = None
isStop = False
isRecording = None
instrument_code = '0'
instrument_path = None
min_octave = None
max_octave = None 
octave_code = None
gesture_preset = '1'
pose_preset = '1'

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
audio = pyaudio.PyAudio()
stream = None
frames = []

play_time = None

image_status = {}

code = {
    '0':'c_low', '1':'d', '2':'e', '3':'f', '4':'g',
    '5':'a', '6':'b', '7':'stop', '8': 'octaveup', '9': 'octavedown'
}

name_code = {
    '0':'도', '1':'레', '2':'미', '3':'파', '4':'솔',
    '5':'라', '6':'시', '7':'정지 신호', '8': '옥타브 업 신호', '9': '옥타브 다운 신호'
}

mode = {
    '0': 'gesture', '1': 'pose'
}

instrument = {
    '0': "piano",
    '1': "pipe",
    '2': "harmonium",
    '3': "xylophone",
    '4': "flute"
}

octave = {
    '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7'
}

def find_octave_range(instrument_path):
    """악기 폴더 내에서 최소 및 최대 옥타브 찾기"""
    octave_names = [name for name in os.listdir(instrument_path) if os.path.isdir(os.path.join(instrument_path, name))]
    octave_numbers = [int(name) for name in octave_names]
    min_octave, max_octave = min(octave_numbers), max(octave_numbers)
    return min_octave, max_octave

def load_sound_file(instrument_path, note):
    # 파일 확장자 우선 순위 정의
    file_extensions = ['ogg', 'wav']
    for ext in file_extensions:
        # 파일 경로 생성
        file_path = f'{instrument_path}/{note}.{ext}'
        # 해당 파일이 존재하는지 확인
        if os.path.exists(file_path):
            # 파일이 존재하면 pygame.mixer.Sound 객체 리턴
            return pygame.mixer.Sound(file_path)
    # 파일이 어떤 확장자로도 존재하지 않는 경우 None 리턴
    print(f"File not found: {file_path}")
    return None

instrument_path = f'flask_server/instrument/{instrument[instrument_code]}'
min_octave, max_octave = find_octave_range(instrument_path)
octave_code = min_octave
instrument_path = f'flask_server/instrument/{instrument[instrument_code]}/{octave_code}'

# 악기안에 번호로 숫자 폴더를 만들어 음역 대 구분
gesture_sounds1 = {
    0: load_sound_file(instrument_path, 'c'),
    1: load_sound_file(instrument_path, 'd'),
    2: load_sound_file(instrument_path, 'e'),
    3: load_sound_file(instrument_path, 'f'),
    4: load_sound_file(instrument_path, 'g'),
    5: load_sound_file(instrument_path, 'a'),
    6: load_sound_file(instrument_path, 'b')
}

gesture_sounds2 = {
    0: load_sound_file(instrument_path, 'c'),
    1: load_sound_file(instrument_path, 'd'),
    2: load_sound_file(instrument_path, 'e'),
    3: load_sound_file(instrument_path, 'f'),
    4: load_sound_file(instrument_path, 'g'),
    5: load_sound_file(instrument_path, 'a'),
    6: load_sound_file(instrument_path, 'b')
}

pose_sounds = {
    0: load_sound_file(instrument_path, 'c'),
    1: load_sound_file(instrument_path, 'd'),
    2: load_sound_file(instrument_path, 'e'),
    3: load_sound_file(instrument_path, 'f'),
    4: load_sound_file(instrument_path, 'g'),
    5: load_sound_file(instrument_path, 'a'),
    6: load_sound_file(instrument_path, 'b')
}

# 현재 날짜 및 시간을 포맷에 맞게 가져오기
current_time = None
date_time = None

def get_sorted_videos(sort_by, sort_direction):
    sort_key = 'metadata.name'  # 기본적으로 이름을 기준으로 정렬합니다.
    if sort_by in ['length', 'creationDate', 'instrument']:
        sort_key = f"metadata.{sort_by}"
    
    sort_order = pymongo.DESCENDING if sort_direction == 'desc' else pymongo.ASCENDING
    
    videos = list(fs.find().sort(sort_key, sort_order))
    
    return videos

def update_current_time():
    global current_time, date_time
      # 현재 시간을 글로벌 변수인 date_time, current_time에 저장. 현재 시간을 "%Y-%m-%d_%H-%M-%S"의 형식으로 포맷.
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def record_audio():
    global isRecording, frames, current_time

    p = pyaudio.PyAudio()

    # 녹음 설정: 체널 수, 샘플 레이트, 버퍼 크기 등을 설정
    stream = p.open(format=pyaudio.paInt16,
                    channels=2,
                    rate=44100,
                    input=True,
                    input_device_index=1,
                    frames_per_buffer=1024)

    frames = [] # 오디오 프레임을 저장할 리스트

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

#2개의 영상을 가로로 합치는 함수
def merge_videos_horizontally(input_video1, input_video2, output_video):
    # FFmpeg 명령어 구성
    command = [
        'ffmpeg',
        '-i', input_video1,
        '-i', input_video2,
        '-c:v', 'libx264',
        '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
        '-map', '[v]',
        output_video
    ]

    try:
        # subprocess를 사용하여 명령어 실행
        subprocess.run(command, check=True)
        print(f"비디오 병합이 성공적으로 완료되었습니다: {output_video}")
        os.remove(input_video1)
        os.remove(input_video2)
    except subprocess.CalledProcessError as e:
        print(f"명령어 실행 중 에러가 발생했습니다: {e}")
    except FileNotFoundError:
        print("FFmpeg가 설치되어 있지 않거나 시스템의 PATH에 추가되어 있지 않습니다.")
    except Exception as e:
        print(f"예상치 못한 에러가 발생했습니다: {e}")

# 오디오 파일과 비디오 파일을 병합하는 함수
def merge_audio_video(video_file, audio_file, output_file): # 오디오, 비디오 파일 병합 함수
    global play_time, date_time
    command = ['ffmpeg', '-y', '-i', video_file, '-i', audio_file, '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', output_file]
    try:
        # 외부 명령어 실행. 병합 과정에서 발생하는 표준 출력과 오류는 각각 stdout, stderr에 저장.
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 동영상 파일을 로드합니다.
        video = VideoFileClip(output_file)
        # 동영상의 재생시간을 구합니다 (초 단위).
        duration = math.floor(video.duration)
        # VideoFileClip 객체를 닫습니다.
        video.close()

        with open(output_file, "rb") as record_file:
            # MongoDB에 저장
            fs.put(record_file, metadata={"name": output_file, "creationDate": date_time, "instrument": instrument[instrument_code], "length": duration})

        # 병합 완료되면 원본 비디오 파일과 오디오 파일을 삭제
        os.remove(video_file)
        os.remove(audio_file)
        os.remove(output_file)
        print(f"Deleted original files: {video_file} and {audio_file}")
    except subprocess.CalledProcessError as e:
        print("Error Occurred:", e)
        print("Error Output:", e.stderr.decode())

# 데이터셋 디렉토리 생성
data_directory = "flask_server/data"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# 악기 디렉토리 생성
instrument_directory = "flask_server/instrument"
if not os.path.exists(instrument_directory):
    os.makedirs(instrument_directory)

# 녹화 디렉토리 생성
output_directory = "flask_server/recordings" 
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def update_gesture_sounds1(octave_code='4'):
    global instrument_code, gesture_sounds1
    instrument_path = f'flask_server/instrument/{instrument[instrument_code]}/{octave_code}'
    gesture_sounds1 = {
        0: load_sound_file(instrument_path, 'c'),
        1: load_sound_file(instrument_path, 'd'),
        2: load_sound_file(instrument_path, 'e'),
        3: load_sound_file(instrument_path, 'f'),
        4: load_sound_file(instrument_path, 'g'),
        5: load_sound_file(instrument_path, 'a'),
        6: load_sound_file(instrument_path, 'b')
    }

def update_gesture_sounds2(octave_code='4'):
    global instrument_code, gesture_sounds2
    instrument_path = f'flask_server/instrument/{instrument[instrument_code]}/{octave_code}'
    gesture_sounds2 = {
        0: load_sound_file(instrument_path, 'c'),
        1: load_sound_file(instrument_path, 'd'),
        2: load_sound_file(instrument_path, 'e'),
        3: load_sound_file(instrument_path, 'f'),
        4: load_sound_file(instrument_path, 'g'),
        5: load_sound_file(instrument_path, 'a'),
        6: load_sound_file(instrument_path, 'b')
    }

def update_pose_sounds(octave_code='4'):
    global instrument_code, pose_sounds
    instrument_path = f'flask_server/instrument/{instrument[instrument_code]}/{octave_code}'
    pose_sounds = {
        0: load_sound_file(instrument_path, 'c'),
        1: load_sound_file(instrument_path, 'd'),
        2: load_sound_file(instrument_path, 'e'),
        3: load_sound_file(instrument_path, 'f'),
        4: load_sound_file(instrument_path, 'g'),
        5: load_sound_file(instrument_path, 'a'),
        6: load_sound_file(instrument_path, 'b')
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
    global gesture_code, gesture_preset
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
        if(isStop) :
            np.savetxt('flask_server/data/gesture/'+ gesture_preset +'/gesture_train_' + code[gesture_code] + '.csv', file, delimiter=',')
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
def get_pose_set():
    global pose_code, pose_preset
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
        if(isStop) :
            np.savetxt('flask_server/data/pose/'+ pose_preset +'/pose_angle_train_' + code[pose_code] + '.csv', file, delimiter=',')
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
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

def gesture_gen():
    global gesture_preset, gesture_sounds1, isRecording, out, height, width, fps, octave_code, min_octave, max_octave
    current_gesture_preset = gesture_preset
    knn = load_and_train_knn(current_gesture_preset)

    max_num_hands = 1

    frame_count = 0
    start_time = time.time()  # 시작 시간 기록

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
        if gesture_preset != current_gesture_preset:
            current_gesture_preset = gesture_preset
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
                        elif idx ==8:
                            octave_code += 1
                            if octave_code > max_octave:  # 여기서 2는 octave_code의 최대값
                                octave_code = max_octave  # 최대값 초과 시 octave_code를 최대값으로 설정
                            octave_code_str = str(octave_code)
                            update_gesture_sounds1(octave_code_str)
                        elif idx == 9:
                            octave_code -= 1
                            if octave_code < min_octave:  # 여기서 2는 octave_code의 최대값
                                octave_code = min_octave  # 최대값 초과 시 octave_code를 최대값으로 설정
                            octave_code_str = str(octave_code)
                            update_gesture_sounds1(octave_code_str)
                        elif idx == 13:
                            pygame.mixer.music.stop()
                        temp_idx = idx
        if isRecording:
            out.write(img)
        # 프레임에 주사율 표시
        cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
# 웹캠
def gesture_gen_2():
    global gesture_preset, gesture_sounds2, isRecording, out2, height, width, fps, octave_code, min_octave, max_octave
    octave_code=5
    current_gesture_preset = gesture_preset
    knn = load_and_train_knn(current_gesture_preset)

    max_num_hands = 1

    frame_count = 0
    start_time = time.time()  # 시작 시간 기록

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
        if gesture_preset != current_gesture_preset:
            current_gesture_preset = gesture_preset
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
                            octave_code += 1
                            if octave_code > max_octave:  # 여기서 2는 octave_code의 최대값
                                octave_code = max_octave  # 최대값 초과 시 octave_code를 최대값으로 설정
                            octave_code_str = str(octave_code)
                            update_gesture_sounds2(octave_code_str)
                        elif idx == 9:
                            octave_code -= 1
                            if octave_code < min_octave:  # 여기서 2는 octave_code의 최대값
                                octave_code = min_octave  # 최대값 초과 시 octave_code를 최대값으로 설정
                            octave_code_str = str(octave_code)
                            update_gesture_sounds2(octave_code_str)
                        elif idx == 13:
                            pygame.mixer.music.stop()
                        else:
                            print(f"{instrument_path}")
                        temp_idx = idx
        if isRecording:
            out2.write(img2)

        # 프레임에 주사율 표시
        cv2.putText(img2, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret2, jpeg = cv2.imencode('.jpg', img2)
        frame2 = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

def pose_gen():
    global pose_preset, pose_sounds, isRecording, out, height, width, fps
    octave_code = 5

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

    current_pose_preset = pose_preset
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
        if pose_preset != current_pose_preset:
            current_pose_preset = pose_preset
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
                        sound.set_volume(0.3)
                        sound.play()
                    elif idx == 8:
                        octave_code += 1
                        if octave_code > 7:  # 여기서 2는 octave_code의 최대값
                            octave_code = 7  # 최대값 초과 시 octave_code를 최대값으로 설정
                        octave_code_str = str(octave_code)
                        update_pose_sounds(octave_code_str)
                    elif idx == 9:
                        octave_code -= 1
                        if octave_code < 1:  # 여기서 2는 octave_code의 최대값
                            octave_code = 1  # 최대값 초과 시 octave_code를 최대값으로 설정
                        octave_code_str = str(octave_code)
                        update_pose_sounds(octave_code_str)
                    elif idx == 13:
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.stop()
            if isRecording:
                out.write(img)
            # 프레임에 주사율 표시
            cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/Get_AllMovements', methods=['GET', 'POST'])
def process_movement_data():
    global gesture_code, gesture_preset, pose_code, pose_preset, isStop, name_code, image_status
    response_data = {'data': []}
    timestamp = {}
    for n_code in name_code:
        image_path = f'flask_server/static/capture/gesture/{gesture_preset}/capture_gesture_'+code[n_code]+'.jpg'
        image_status[n_code] = os.path.exists(image_path)
        timestamp[n_code] = int(os.path.getmtime(image_path)) if image_status[n_code] else ""
    if request.method == 'POST':        
        mode = request.form.get('mode')
        if (mode == "gesture"):
            response_data = {'data': []}
            for n_code in name_code:
                image_path = f'flask_server/static/capture/gesture/{gesture_preset}/capture_gesture_'+code[n_code]+'.jpg'
                image_status[n_code] = os.path.exists(image_path)
                timestamp[n_code] = int(os.path.getmtime(image_path)) if image_status[n_code] else ""
                response_data['data'].append({
                    'idx': n_code,
                    'image_status': image_status[n_code],
                    'image_url': f"static/capture/gesture/{gesture_preset}/capture_gesture_{code[n_code]}.jpg",
                    'timestamp': timestamp[n_code]
                })
            if 'preset' in request.form:
                image_status = {}
                response_data = {'data': []}
                gesture_preset = request.form['preset']
                for n_code in name_code:
                    image_path = f'flask_server/static/capture/gesture/{gesture_preset}/capture_gesture_'+code[n_code]+'.jpg'
                    image_status[n_code] = os.path.exists(image_path)
                    timestamp[n_code] = int(os.path.getmtime(image_path)) if image_status[n_code] else ""
                    response_data['data'].append({
                        'idx': n_code,
                        'image_status': image_status[n_code],
                        'image_url': f"static/capture/gesture/{gesture_preset}/capture_gesture_{code[n_code]}.jpg",
                        'timestamp': timestamp[n_code]
                    })
                return jsonify(response_data)
                        
            elif 'button_value' in request.form:
                gesture_code = request.form['button_value']
                session['button_value_received'] = True  # 세션에 상태 저장          
                return jsonify(response_data)

            elif 'stop_sign' in request.form and session.get('button_value_received'):
                isStop = request.form['stop_sign']
                session['button_value_received'] = False  # 상태 초기화
                image_status = {}
                response_data = {'data': []}
                for n_code in name_code:
                    image_path = f'flask_server/static/capture/gesture/{gesture_preset}/capture_gesture_'+code[n_code]+'.jpg'
                    image_status[n_code] = os.path.exists(image_path)
                    timestamp[n_code] = int(os.path.getmtime(image_path)) if image_status[n_code] else ""
                    response_data['data'].append({
                        'idx': n_code,
                        'image_status': image_status[n_code],
                        'image_url': f"static/capture/gesture/{gesture_preset}/capture_gesture_{code[n_code]}.jpg",
                        'timestamp': timestamp[n_code]
                    })
                return jsonify(response_data)

            elif 'delete_button_value' in request.form:
                gesture_code = request.form['delete_button_value']
                file_path = 'flask_server/data/gesture/' + gesture_preset + '/gesture_train_' + code[gesture_code] + '.csv'

                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"파일 삭제 중 오류 발생: {e}")
                file_path = 'flask_server/static/capture/gesture/' + gesture_preset + '/capture_gesture_' + code[gesture_code] + '.jpg'

                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"파일 삭제 중 오류 발생: {e}")
                image_status = {}
                response_data = {'data': []}
                for n_code in name_code:
                    image_path = f'flask_server/static/capture/gesture/{gesture_preset}/capture_gesture_'+code[n_code]+'.jpg'
                    image_status[n_code] = os.path.exists(image_path)
                    timestamp[n_code] = int(os.path.getmtime(image_path)) if image_status[n_code] else ""
                    response_data['data'].append({
                        'idx': n_code,
                        'image_status': image_status[n_code],
                        'image_url': f"static/capture/gesture/{gesture_preset}/capture_gesture_{code[n_code]}.jpg",
                        'timestamp': timestamp[n_code]
                    })
                return jsonify(response_data)
            
            return jsonify(response_data)
            
        if (mode == "pose"):
            image_status = {}
            response_data = {'data': []}
            for n_code in name_code:
                image_path = f'flask_server/static/capture/pose/{pose_preset}/capture_pose_'+code[n_code]+'.jpg'
                image_status[n_code] = os.path.exists(image_path)
                timestamp[n_code] = int(os.path.getmtime(image_path)) if image_status[n_code] else ""
                response_data['data'].append({
                    'idx': n_code,
                    'image_status': image_status[n_code],
                    'image_url': f"static/capture/pose/{pose_preset}/capture_pose_{code[n_code]}.jpg",
                    'timestamp': timestamp[n_code]
                })
            if 'preset' in request.form:
                image_status = {}
                response_data = {'data': []}
                pose_preset = request.form['preset']
                for n_code in name_code:
                    image_path = f'flask_server/static/capture/pose/{pose_preset}/capture_pose_'+code[n_code]+'.jpg'
                    image_status[n_code] = os.path.exists(image_path)
                    timestamp[n_code] = int(os.path.getmtime(image_path)) if image_status[n_code] else ""
                    response_data['data'].append({
                        'idx': n_code,
                        'image_status': image_status[n_code],
                        'image_url': f"static/capture/pose/{pose_preset}/capture_pose_{code[n_code]}.jpg",
                        'timestamp': timestamp[n_code]
                    })
                return jsonify(response_data)

            elif 'button_value' in request.form:
                pose_code = request.form['button_value']
                session['button_value_received'] = True  # 세션에 상태 저장
                return jsonify(response_data)

            elif 'stop_sign' in request.form and session.get('button_value_received'):
                isStop = request.form['stop_sign']
                session['button_value_received'] = False  # 상태 초기화
                image_status = {}
                response_data = {'data': []}
                for n_code in name_code:
                    image_path = f'flask_server/static/capture/pose/{pose_preset}/capture_pose_'+code[n_code]+'.jpg'
                    image_status[n_code] = os.path.exists(image_path)
                    timestamp[n_code] = int(os.path.getmtime(image_path)) if image_status[n_code] else ""
                    response_data['data'].append({
                        'idx': n_code,
                        'image_status': image_status[n_code],
                        'image_url': f"static/capture/pose/{pose_preset}/capture_pose_{code[n_code]}.jpg",
                        'timestamp': timestamp[n_code]
                    })
                return jsonify(response_data)
            
            elif 'delete_button_value' in request.form:
                pose_code = request.form['delete_button_value']
                file_path = 'flask_server/data/pose/' + pose_preset + '/pose_angle_train_' + code[pose_code] + '.csv'
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"파일 삭제 중 오류 발생: {e}")
                file_path = 'flask_server/static/capture/pose/' + pose_preset + '/capture_pose_' + code[pose_code] + '.jpg'

                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"파일 삭제 중 오류 발생: {e}")
                image_status = {}
                response_data = {'data': []}
                for n_code in name_code:
                    image_path = f'flask_server/static/capture/pose/{pose_preset}/capture_pose_'+code[n_code]+'.jpg'
                    image_status[n_code] = os.path.exists(image_path)
                    timestamp[n_code] = int(os.path.getmtime(image_path)) if image_status[n_code] else ""
                    response_data['data'].append({
                        'idx': n_code,
                        'image_status': image_status[n_code],
                        'image_url': f"static/capture/pose/{pose_preset}/capture_pose_{code[n_code]}.jpg",
                        'timestamp': timestamp[n_code]
                    })
                return jsonify(response_data)

            return jsonify(response_data)
    return render_template('GetMovementDataSet.html',timestamp=timestamp, mode="gesture", code=code, image_status=image_status, gesture_preset=gesture_preset, pose_preset=pose_preset, name_codes=name_code)

@app.route('/reset', methods=['POST'])
def reset():
    global gesture_preset, pose_preset

    gesture_preset = '1'
    pose_preset = '1'

    for i in range(1, 5):
        # 기존에 수집된 데이터셋 초기화
        gesture_file_path = 'flask_server/data/gesture/'+ str(i) +'/gesture_train.csv'
        pose_file_path = 'flask_server/data/pose/'+ str(i) +'/pose_angle_train.csv'
        if os.path.exists(gesture_file_path):
            os.remove(gesture_file_path)
        if os.path.exists(pose_file_path):
            os.remove(pose_file_path)
        if(i == 1) :
            for j in range(0, 10) :
                # 목표 CSV 파일이 없으면 새로 생성
                if not os.path.exists('flask_server/data/gesture/'+ str(i) +'/gesture_train_'+ code[str(j)] + '.csv'):
                    open('flask_server/data/gesture/'+ str(i) +'/gesture_train_'+ code[str(j)] + '.csv', 'w').close()

                # 원본 CSV 파일 읽기
                with open('flask_server/data/gesture/원본/gesture_train_'+ code[str(j)] + '.csv', 'r', newline='', encoding='utf-8') as source_file:
                    reader = csv.reader(source_file)
                    data = list(reader)
        
                # 목표 CSV 파일에 쓰기
                with open('flask_server/data/gesture/'+ str(i) +'/gesture_train_'+ code[str(j)] + '.csv', 'w', newline='', encoding='utf-8') as target_file:
                    writer = csv.writer(target_file)
                    writer.writerows(data)

                # 목표 JPG 파일이 없으면 새로 생성
                if not os.path.exists('flask_server/static/capture/gesture/'+ str(i) +'/capture_gesture_'+ code[str(j)] + '.jpg'):
                    open('flask_server/static/capture/gesture/'+ str(i) +'/capture_gesture_'+ code[str(j)] + '.jpg', 'w').close()

                # 원본 JPG 파일 복사
                if os.path.exists('flask_server/static/capture/gesture/'+ str(i) +'/capture_gesture_'+ code[str(j)] + '.jpg'):
                    with Image.open('flask_server/static/capture/gesture/원본/capture_gesture_'+ code[str(j)] + '.jpg') as img:
                        img.save('flask_server/static/capture/gesture/1/capture_gesture_'+ code[str(j)] + '.jpg')

                if not os.path.exists('flask_server/static/capture/pose/'+ str(i) +'/capture_pose_'+ code[str(j)] + '.jpg'):
                    open('flask_server/static/capture/pose/'+ str(i) +'/capture_pose_'+ code[str(j)] + '.jpg', 'w').close()
                    
                # 원본 JPG 파일 복사
                if os.path.exists('flask_server/static/capture/pose/'+ str(i) +'/capture_pose_'+ code[str(j)] + '.jpg'):
                    with Image.open('flask_server/static/capture/pose/원본/capture_pose_'+ code[str(j)] + '.jpg') as img:
                        img.save('flask_server/static/capture/pose/1/capture_pose_'+ code[str(j)] + '.jpg')
        else :
            # 디렉터리 내의 모든 파일을 반복
            for filename in os.listdir('flask_server/data/gesture/' + str(i)):
                gesture_file_path = os.path.join('flask_server/data/gesture/' + str(i), filename)
                try:
                    if os.path.isfile(gesture_file_path) or os.path.islink(gesture_file_path):
                        os.unlink(gesture_file_path)  # 파일 또는 링크 삭제
                except Exception as e:
                    return jsonify({'status': 'error', 'message': str(e)})
            for filename in os.listdir('flask_server/data/pose/' + str(i)):
                pose_file_path = os.path.join('flask_server/data/pose/' + str(i), filename)
                try:
                    if os.path.isfile(pose_file_path) or os.path.islink(pose_file_path):
                        os.unlink(pose_file_path)  # 파일 또는 링크 삭제
                except Exception as e:
                    return jsonify({'status': 'error', 'message': str(e)})
            for filename in os.listdir('flask_server/static/capture/gesture/' + str(i)):
                gesture_file_path = os.path.join('flask_server/static/capture/gesture/' + str(i), filename)
                try:
                    if os.path.isfile(gesture_file_path) or os.path.islink(gesture_file_path):
                        os.unlink(gesture_file_path)  # 파일 또는 링크 삭제
                except Exception as e:
                    return jsonify({'status': 'error', 'message': str(e)})
            for filename in os.listdir('flask_server/static/capture/pose/' + str(i)):
                pose_file_path = os.path.join('flask_server/static/capture/pose/' + str(i), filename)
                try:
                    if os.path.isfile(pose_file_path) or os.path.islink(pose_file_path):
                        os.unlink(pose_file_path)  # 파일 또는 링크 삭제
                except Exception as e:
                    return jsonify({'status': 'error', 'message': str(e)})    
    image_status = {}
    response_data = {'data': []}
    timestamp = {}
    for n_code in name_code:
        image_path = f'flask_server/static/capture/gesture/1/capture_gesture_'+code[n_code]+'.jpg'
        image_status[n_code] = os.path.exists(image_path)
        # 가상의 데이터 생성 예제
        timestamp[n_code] = int(os.path.getmtime(image_path)) if image_status[n_code] else ""
        response_data['data'].append({
            'idx': n_code,
            'image_status': image_status[n_code],
            'image_url': f"static/capture/gesture/1/capture_gesture_{code[n_code]}.jpg",
            'timestamp': timestamp[n_code]
        })
    return jsonify(response_data)

@app.route('/Movement_play', methods=['GET', 'POST'])
def hand_gestures_play():
    global instrument_code, gesture_preset, pose_preset, isRecording, fourcc, out, out2, audio_recording_thread, current_time, fps, width, height
    if request.method == 'POST':
        mode = request.form.get('mode')
        if (mode == "gesture"):
            if 'preset' in request.form:
                gesture_preset = request.form['preset']
                return render_template('MovementPlay.html', message="프리셋 변경")
            if 'isRecording' in request.form:
                if(request.form['isRecording'] == 'True') :
                    update_current_time()
                    out = cv2.VideoWriter(f"{output_directory}/output_{current_time}_left.avi", fourcc, fps, (width, height))
                    out2 = cv2.VideoWriter(f"{output_directory}/output_{current_time}_right.avi", fourcc, fps, (width, height))
                    isRecording = True
                    audio_recording_thread = threading.Thread(target=record_audio)
                    audio_recording_thread.start()
                    return render_template('MovementPlay.html', message="녹화를 시작합니다.")
                elif(request.form['isRecording'] == 'False') :
                    isRecording = None
                    audio_recording_thread.join()
                    out.release()
                    out2.release()
                    merge_videos_horizontally(f"{output_directory}/output_{current_time}_left.avi", f"{output_directory}/output_{current_time}_right.avi", f"{output_directory}/output_{current_time}.avi")
                    merge_audio_video(f"{output_directory}/output_{current_time}.avi", 
                                    f"{output_directory}/output_{current_time}.wav", f"final_output_{current_time}.mp4")
                    return render_template('MovementPlay.html', message="녹화를 종료합니다.")
        if (mode == "pose"):
            if 'preset' in request.form:
                pose_preset = request.form['preset']
                return render_template('MovementPlay.html', message="프리셋 변경")
            if 'isRecording' in request.form:
                if(request.form['isRecording'] == 'True') :
                    update_current_time()
                    out = cv2.VideoWriter(f"{output_directory}/output_{current_time}.avi", fourcc, fps, (width, height))
                    isRecording = True
                    audio_recording_thread = threading.Thread(target=record_audio)
                    audio_recording_thread.start()
                    return render_template('MovementPlay.html', message="녹화를 시작합니다.")
                elif(request.form['isRecording'] == 'False') :
                    isRecording = None
                    audio_recording_thread.join()
                    out.release()
                    merge_audio_video(f"{output_directory}/output_{current_time}.avi", 
                                    f"{output_directory}/output_{current_time}.wav", f"final_output_{current_time}.mp4")
                    return render_template('MovementPlay.html', message="녹화를 종료합니다.")
    return render_template('MovementPlay.html', gesture_preset=gesture_preset, pose_preset=pose_preset)

@app.route('/Instrument_choice', methods=['GET', 'POST'])
def instrument_choice():
    global instrument_code, instrument_path, min_octave, max_octave, octave_code
    if request.method == 'POST':
        if 'instrument_value' in request.form:
            instrument_code = request.form['instrument_value']
            instrument_path = f'flask_server/instrument/{instrument[instrument_code]}'
            min_octave, max_octave = find_octave_range(instrument_path)
            octave_code = min_octave
            update_gesture_sounds1(octave_code)
            update_gesture_sounds2(octave_code)
            update_pose_sounds(octave_code)
            return render_template('instrumentChoice.html', message="악기 변경")
    return render_template('instrumentChoice.html')

@app.route('/Playlist')   
def playlist():
    global sort_by, sort_direction
    # MongoDB에서 영상 리스트 가져오기
    videolist = list(fs.find())  # GridFS에서 모든 파일을 조회
    return render_template('Playlist.html', videos=videolist, sort_by=sort_by, sort_direction=sort_direction)

@app.route('/Playlist/view/<video_id>', methods=['GET'])
def view(video_id):
    # MongoDB에서 해당 비디오 파일 가져오기
    video = fs.find_one({'_id': bson.ObjectId(video_id)})
    # 비디오 데이터를 스트리밍하는 응답 생성
    def generate():
        for chunk in video:
            yield chunk
    if video:
        return Response(generate(), mimetype='video/mp4')
    else:
        return 'File not found', 404

@app.route('/Playlist/rename/<video_id>', methods=['POST'])
def rename(video_id):
    new_name = request.form.get('new_name')
    try:
        # MongoDB에서 해당 비디오ID를 가진 문서의 metadata.name을 업데이트
        result = fs_files_collection.update_one(
            {'_id': ObjectId(video_id)},
            {'$set': {'metadata.name': new_name}}
        )
        # 문서를 찾지 못한 경우
        if result.matched_count == 0:
            return jsonify({'success': False, 'message': 'File not found'}), 404         
        # 성공적으로 업데이트
        return jsonify({'success': True, 'message': 'Video name updated successfully.'}), 200
    except Exception as e:
        error_message = str(e) if str(e) else 'An error occurred while updating video name.'
        return jsonify({'success': False, 'message': error_message}), 500

@app.route('/Playlist/delete_selected', methods=['POST'])
def delete_selected():
    video_ids_raw = request.form.getlist('video_ids')
    sort_by = request.form.get('sort_by', 'creationDate')
    sort_direction = request.form.get('sort_direction', 'asc')

    video_ids = []
    for video_id in video_ids_raw:
        video_ids.extend(video_id.split(','))
    
    for video_id in video_ids:
        fs.delete(ObjectId(video_id))
    return redirect(url_for('list_videos', sort_by=sort_by, sort_direction=sort_direction))

@app.route('/Playlist/<sort_by>/<sort_direction>')
def list_videos(sort_by, sort_direction):
    videolist = get_sorted_videos(sort_by, sort_direction)
    # HTML 템플릿에 비디오 목록 전달
    return render_template('Playlist.html', videos=videolist, sort_by=sort_by, sort_direction=sort_direction)

@app.route('/processed_video_gesture')
def processed_video_gesture():
    return Response(gesture_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_video_gesture_2')
def processed_video_gesture_2():
    return Response(gesture_gen_2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_video_pose')
def processed_video_pose():
    return Response(pose_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_video_gesture')
def get_video_gesture():
    return Response(get_gesture_set(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_video_pose')
def get_video_pose():
    return Response(get_pose_set(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/music_files/<instrument_code>')
def get_music_files(instrument_code):
    instrument_dir = instrument.get(instrument_code)
    if not instrument_dir:
        return jsonify([]), 404
    music_dir = os.path.join('flask_server/preview_sound', instrument_dir)
    files = [f for f in os.listdir(music_dir) if f.endswith('.ogg') or f.endswith('.wav')]
    return jsonify(files)
    
@app.route('/music/<instrument_code>/<filename>')
def serve_music_file(instrument_code, filename):
    instrument_dir = instrument.get(instrument_code)
    if not instrument_dir:
        return '', 404
    music_dir = os.path.join('preview_sound', instrument_dir)
    return send_from_directory(music_dir, filename)

@app.route('/')
def get_main():
    return render_template('MovementPlay.html')

if __name__ == '__main__':
    app.run(threaded=True, debug=True)