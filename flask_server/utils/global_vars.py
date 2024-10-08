# utils/global_vars.py
from pymongo import MongoClient
from gridfs import GridFS
import cv2, datetime, os, pymongo, pygame

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"]="0"

client = MongoClient("mongodb://localhost:27017/")
db = client["record_videos"]
fs = GridFS(db)
fs_files_collection = db['fs.files']  # GridFS의 메타데이터 컬렉션

pygame.init()
pygame.mixer.init()
pygame.mixer.set_num_channels(16)

# 전역 변수를 저장하는 딕셔너리
global_vars = {
    'gesture_preset': '1',
    'pose_preset': '1',
    'isStop' : False,
    'isRecording': None,
    'instrument_code': '0',
    'left_instrument_path': 'flask_server/instrument/piano/4',
    'right_instrument_path': 'flask_server/instrument/piano/5',
    'pose_instrument_path': 'flask_server/instrument/piano/4',
    'fourcc': cv2.VideoWriter_fourcc(*'XVID'),
    'out': None,
    'out2': None,
    'current_time': None,
    'date_time': None,
    'fps': 30.0,
    'width': 640,
    'height': 480,
    'play_time': None,
    'min_octave': 1,
    'max_octave': 7, 
    'gesture_left_octave_code': 4,
    'gesture_right_octave_code': 4,
    'pose_octave_code': 4,
    'gesture_code': None,
    'pose_code': None
}

# 전역 변수에 접근하고 수정하는 함수
def get_global_var(key):
    return global_vars.get(key)

def set_global_var(key, value):
    global_vars[key] = value

    # instrument_path 변경 시 사운드 파일 재로딩
    if key == 'left_instrument_path':
        reload_sounds()
    elif key == 'right_instrument_path':
        reload_sounds()
    elif key == 'pose_instrument_path':
        reload_sounds()

def update_current_time():
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    set_global_var('current_time', current_time)
    set_global_var('date_time', date_time)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap2 = cv2.VideoCapture(1, cv2.CAP_MSMF)

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640), cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480), cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30.0), cap2.set(cv2.CAP_PROP_FPS, 30.0)

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

def get_sorted_videos(sort_by, sort_direction):
    sort_key = 'metadata.name'  # 기본적으로 이름을 기준으로 정렬합니다.
    if sort_by in ['length', 'creationDate', 'instrument']:
        sort_key = f"metadata.{sort_by}"
    
    sort_order = pymongo.DESCENDING if sort_direction == 'desc' else pymongo.ASCENDING
    
    videos = list(fs.find().sort(sort_key, sort_order))
    
    return videos

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

# 악기안에 번호로 숫자 폴더를 만들어 음역 대 구분
gesture_sounds1 = {
    0: load_sound_file(get_global_var('left_instrument_path'), 'c'),
    1: load_sound_file(get_global_var('left_instrument_path'), 'd'),
    2: load_sound_file(get_global_var('left_instrument_path'), 'e'),
    3: load_sound_file(get_global_var('left_instrument_path'), 'f'),
    4: load_sound_file(get_global_var('left_instrument_path'), 'g'),
    5: load_sound_file(get_global_var('left_instrument_path'), 'a'),
    6: load_sound_file(get_global_var('left_instrument_path'), 'b')
}

gesture_sounds2 = {
    0: load_sound_file(get_global_var('right_instrument_path'), 'c'),
    1: load_sound_file(get_global_var('right_instrument_path'), 'd'),
    2: load_sound_file(get_global_var('right_instrument_path'), 'e'),
    3: load_sound_file(get_global_var('right_instrument_path'), 'f'),
    4: load_sound_file(get_global_var('right_instrument_path'), 'g'),
    5: load_sound_file(get_global_var('right_instrument_path'), 'a'),
    6: load_sound_file(get_global_var('right_instrument_path'), 'b')
}

pose_sounds = {
    0: load_sound_file(get_global_var('pose_instrument_path'), 'c'),
    1: load_sound_file(get_global_var('pose_instrument_path'), 'd'),
    2: load_sound_file(get_global_var('pose_instrument_path'), 'e'),
    3: load_sound_file(get_global_var('pose_instrument_path'), 'f'),
    4: load_sound_file(get_global_var('pose_instrument_path'), 'g'),
    5: load_sound_file(get_global_var('pose_instrument_path'), 'a'),
    6: load_sound_file(get_global_var('pose_instrument_path'), 'b')
}

def reload_sounds():
    gesture_sounds1.update({
        0: load_sound_file(get_global_var('left_instrument_path'), 'c'),
        1: load_sound_file(get_global_var('left_instrument_path'), 'd'),
        2: load_sound_file(get_global_var('left_instrument_path'), 'e'),
        3: load_sound_file(get_global_var('left_instrument_path'), 'f'),
        4: load_sound_file(get_global_var('left_instrument_path'), 'g'),
        5: load_sound_file(get_global_var('left_instrument_path'), 'a'),
        6: load_sound_file(get_global_var('left_instrument_path'), 'b')
    })

    gesture_sounds2.update({
        0: load_sound_file(get_global_var('right_instrument_path'), 'c'),
        1: load_sound_file(get_global_var('right_instrument_path'), 'd'),
        2: load_sound_file(get_global_var('right_instrument_path'), 'e'),
        3: load_sound_file(get_global_var('right_instrument_path'), 'f'),
        4: load_sound_file(get_global_var('right_instrument_path'), 'g'),
        5: load_sound_file(get_global_var('right_instrument_path'), 'a'),
        6: load_sound_file(get_global_var('right_instrument_path'), 'b')
    })

    pose_sounds.update({
        0: load_sound_file(get_global_var('pose_instrument_path'), 'c'),
        1: load_sound_file(get_global_var('pose_instrument_path'), 'd'),
        2: load_sound_file(get_global_var('pose_instrument_path'), 'e'),
        3: load_sound_file(get_global_var('pose_instrument_path'), 'f'),
        4: load_sound_file(get_global_var('pose_instrument_path'), 'g'),
        5: load_sound_file(get_global_var('pose_instrument_path'), 'a'),
        6: load_sound_file(get_global_var('pose_instrument_path'), 'b')
    })