from flask import Blueprint, render_template, request, Response, jsonify, session
from utils.global_vars import get_global_var, set_global_var, update_current_time, output_directory, code, name_code
from utils.video_utils import merge_videos_horizontally, merge_audio_video
from utils.audio_utils import record_audio
from utils.video_processing import gesture_gen, gesture_gen_2, pose_gen, get_gesture_set, get_pose_set
from PIL import Image
import threading, cv2, os, csv

movement_blueprint = Blueprint('movement', __name__)

@movement_blueprint.route('/Get_AllMovements', methods=['GET', 'POST'])
def process_movement_data():
    gesture_preset = get_global_var('gesture_preset')
    pose_preset = get_global_var('pose_preset')
    response_data = {'data': []}
    timestamp = {}
    image_status = {}
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
                set_global_var('gesture_preset', gesture_preset)
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
                set_global_var('gesture_code', gesture_code)
                session['button_value_received'] = True  # 세션에 상태 저장          
                return jsonify(response_data)

            elif 'stop_sign' in request.form and session.get('button_value_received'):
                isStop = request.form['stop_sign']
                set_global_var('isStop', isStop)
                gesture_preset = get_global_var('gesture_preset')
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
                gesture_preset = get_global_var('gesture_preset')
                gesture_code = request.form['delete_button_value']
                set_global_var('gesture_code', gesture_code)
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
            pose_preset = get_global_var('pose_preset')
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
                set_global_var('pose_preset', pose_preset)
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
                set_global_var('pose_code', pose_code)
                session['button_value_received'] = True  # 세션에 상태 저장
                return jsonify(response_data)

            elif 'stop_sign' in request.form and session.get('button_value_received'):
                isStop = request.form['stop_sign']
                set_global_var('isStop', isStop)
                session['button_value_received'] = False  # 상태 초기화
                image_status = {}
                response_data = {'data': []}
                pose_preset = get_global_var('pose_preset')
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
                set_global_var('pose_code', pose_code)
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
                pose_preset = get_global_var('pose_preset')
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

@movement_blueprint.route('/reset', methods=['POST'])
def reset():
    set_global_var('gesture_preset', '1')
    set_global_var('pose_preset', '1')

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

@movement_blueprint.route('/Movement_play', methods=['GET', 'POST'])
def movement_play():
    global audio_recording_thread
    audio_recording_thread = None 
    gesture_preset = get_global_var('gesture_preset')
    pose_preset = get_global_var('pose_preset')
    if request.method == 'POST':
        mode = request.form.get('mode')
        if mode == "gesture":
            if 'preset' in request.form:
                gesture_preset = request.form['preset']
                set_global_var('gesture_preset', gesture_preset)
                return render_template('MovementPlay.html', message="프리셋 변경")
            
            if 'isRecording' in request.form:
                if request.form['isRecording'] == 'True':
                    update_current_time()
                    current_time = get_global_var('current_time')

                    out = cv2.VideoWriter(f"{output_directory}/output_{current_time}_left.avi", get_global_var('fourcc'), get_global_var('fps'), (get_global_var('width'), get_global_var('height')))
                    set_global_var('out', out)
                    
                    out2 = cv2.VideoWriter(f"{output_directory}/output_{current_time}_right.avi", get_global_var('fourcc'), get_global_var('fps'), (get_global_var('width'), get_global_var('height')))
                    set_global_var('out2', out2)
                    
                    set_global_var('isRecording', True)

                    audio_recording_thread = threading.Thread(target=record_audio)
                    audio_recording_thread.start()
                    
                    return render_template('MovementPlay.html', message="녹화를 시작합니다.")
                
                elif request.form['isRecording'] == 'False':
                    set_global_var('isRecording', None)

                    if audio_recording_thread is not None:
                        audio_recording_thread.join()
                    
                    out = get_global_var('out')
                    out2 = get_global_var('out2')
                    if out is not None:
                        out.release()
                    if out2 is not None:
                        out2.release()
                    set_global_var('out', None)
                    set_global_var('out2', None)
                    
                    current_time = get_global_var('current_time')
                    merge_videos_horizontally(f"{output_directory}/output_{current_time}_left.avi", 
                                            f"{output_directory}/output_{current_time}_right.avi", f"{output_directory}/output_{current_time}.avi")
                    merge_audio_video(f"{output_directory}/output_{current_time}.avi", 
                                    f"{output_directory}/output_{current_time}.wav", f"final_output_{current_time}.mp4")
                    return render_template('MovementPlay.html', message="녹화를 종료합니다.")
        if (mode == "pose"):
            if 'preset' in request.form:
                pose_preset = request.form['preset']
                set_global_var('pose_preset', pose_preset)
                return render_template('MovementPlay.html', message="프리셋 변경")
            if 'isRecording' in request.form:
                if(request.form['isRecording'] == 'True') :
                    update_current_time()
                    current_time = get_global_var('current_time')

                    out = cv2.VideoWriter(f"{output_directory}/output_{current_time}.avi", get_global_var('fourcc'), get_global_var('fps'), (get_global_var('width'), get_global_var('height')))
                    set_global_var('out', out)

                    set_global_var('isRecording', True)

                    audio_recording_thread = threading.Thread(target=record_audio)
                    audio_recording_thread.start()

                    return render_template('MovementPlay.html', message="녹화를 시작합니다.")
                elif(request.form['isRecording'] == 'False') :
                    set_global_var('isRecording', None)

                    if audio_recording_thread is not None:
                        audio_recording_thread.join()

                    out = get_global_var('out')
                    out.release()
                    set_global_var('out', None)

                    current_time = get_global_var('current_time')
                    merge_audio_video(f"{output_directory}/output_{current_time}.avi", 
                                    f"{output_directory}/output_{current_time}.wav", f"final_output_{current_time}.mp4")
                    return render_template('MovementPlay.html', message="녹화를 종료합니다.")
    return render_template('MovementPlay.html', gesture_preset=gesture_preset, pose_preset=pose_preset)

@movement_blueprint.route('/processed_video_gesture')
def processed_video_gesture():
    return Response(gesture_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@movement_blueprint.route('/processed_video_gesture_2')
def processed_video_gesture_2():
    return Response(gesture_gen_2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@movement_blueprint.route('/processed_video_pose')
def processed_video_pose():
    return Response(pose_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@movement_blueprint.route('/get_video_gesture')
def get_video_gesture():
    return Response(get_gesture_set(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@movement_blueprint.route('/get_video_pose')
def get_video_pose():
    return Response(get_pose_set(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')