from moviepy.editor import VideoFileClip #재생 시간 확인 용도
import os, subprocess, math
from utils.global_vars import instrument, get_global_var, fs

def merge_videos_horizontally(left_video, right_video, output_video):
    # FFmpeg 명령어 구성
    command = [
        'ffmpeg',
        '-i', left_video,
        '-i', right_video,
        '-c:v', 'libx264',
        '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
        '-map', '[v]',
        output_video
    ]

    try:
        # subprocess를 사용하여 명령어 실행
        subprocess.run(command, check=True)
        print(f"비디오 병합이 성공적으로 완료되었습니다: {output_video}")
        os.remove(left_video)
        os.remove(right_video)
    except subprocess.CalledProcessError as e:
        print(f"명령어 실행 중 에러가 발생했습니다: {e}")
    except FileNotFoundError:
        print("FFmpeg가 설치되어 있지 않거나 시스템의 PATH에 추가되어 있지 않습니다.")
    except Exception as e:
        print(f"예상치 못한 에러가 발생했습니다: {e}")
    pass

def merge_audio_video(video_path, audio_path, output_path):
    command = ['ffmpeg', '-y', '-i', video_path, '-i', audio_path, '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', output_path]
    try:
        # 외부 명령어 실행. 병합 과정에서 발생하는 표준 출력과 오류는 각각 stdout, stderr에 저장.
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 동영상 파일을 로드합니다.
        video = VideoFileClip(output_path)
        # 동영상의 재생시간을 구합니다 (초 단위).
        duration = math.floor(video.duration)
        # VideoFileClip 객체를 닫습니다.
        video.close()

        with open(output_path, "rb") as record_file:
            # MongoDB에 저장
            fs.put(record_file, metadata={"name": output_path, "creationDate": get_global_var('date_time'), "instrument": instrument[get_global_var('instrument_code')], "length": duration})

        # 병합 완료되면 원본 비디오 파일과 오디오 파일을 삭제
        os.remove(video_path)
        os.remove(audio_path)
        os.remove(output_path)
        print(f"Deleted original files: {video_path} and {audio_path}")
    except subprocess.CalledProcessError as e:
        print("Error Occurred:", e)
        print("Error Output:", e.stderr.decode())
    pass