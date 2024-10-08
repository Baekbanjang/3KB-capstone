import pyaudio, wave, os
from utils.global_vars import output_directory, get_global_var, instrument, set_global_var

def record_audio():
    global frames

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
    while get_global_var('isRecording'):
        data = stream.read(1024)
        frames.append(data)

    # 녹음 종료 과정
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 오디오 파일로 저장
    wf = wave.open(f"{output_directory}/output_{get_global_var('current_time')}.wav", 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()
    frames = [] # 오디오 프레임 데이터를 초기화
    pass

def find_octave_range(instrument_path):
    """악기 폴더 내에서 최소 및 최대 옥타브 찾기"""
    octave_names = [name for name in os.listdir(instrument_path) if os.path.isdir(os.path.join(instrument_path, name))]
    octave_numbers = [int(name) for name in octave_names]
    min_octave, max_octave = min(octave_numbers), max(octave_numbers)
    return min_octave, max_octave

def update_gesture_sounds1(octave_code):
    instrument_code = get_global_var('instrument_code')
    instrument_path = f'flask_server/instrument/{instrument[instrument_code]}/{octave_code}'
    set_global_var('left_instrument_path', instrument_path)
    pass

def update_gesture_sounds2(octave_code):
    instrument_code = get_global_var('instrument_code')
    instrument_path = f'flask_server/instrument/{instrument[instrument_code]}/{octave_code}'
    set_global_var('right_instrument_path', instrument_path)
    pass

def update_pose_sounds(octave_code):
    instrument_code = get_global_var('instrument_code')
    instrument_path = f'flask_server/instrument/{instrument[instrument_code]}/{octave_code}'
    set_global_var('pose_instrument_path', instrument_path)
    pass