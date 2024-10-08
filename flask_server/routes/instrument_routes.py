from flask import Blueprint, jsonify, request, send_from_directory, render_template
from utils.global_vars import set_global_var, instrument
from utils.audio_utils import find_octave_range
import os

instrument_blueprint = Blueprint('instrument', __name__)

# 악기 선택 및 관련 설정
@instrument_blueprint.route('/Instrument_choice', methods=['GET', 'POST'])
def instrument_choice():
    if request.method == 'POST':
        if 'instrument_value' in request.form:
            instrument_code = request.form['instrument_value']
            
            instrument_path = f'flask_server/instrument/{instrument[instrument_code]}'
            min_octave, max_octave = find_octave_range(instrument_path)
            octave_code = int((min_octave + max_octave)/2)
            instrument_path = f'flask_server/instrument/{instrument[instrument_code]}/{octave_code}'
            set_global_var('min_octave', min_octave)
            set_global_var('max_octave', max_octave)
            set_global_var('gesture_left_octave_code', octave_code)  # 왼쪽 제스처 옥타브 설정
            set_global_var('gesture_right_octave_code', octave_code)  # 오른쪽 제스처 옥타브 설정
            set_global_var('pose_octave_code', octave_code)  # 포즈 옥타브 설정
            set_global_var('instrument_code', instrument_code)
            set_global_var('left_instrument_path', instrument_path)
            set_global_var('right_instrument_path', instrument_path)
            set_global_var('pose_instrument_path', instrument_path)
            
            return render_template('instrumentChoice.html', message="악기 변경")    
    return render_template('instrumentChoice.html')

@instrument_blueprint.route('/api/music_files/<instrument_code>')
def get_music_files(instrument_code):
    instrument_dir = instrument.get(instrument_code)
    if not instrument_dir:
        return jsonify([]), 404
    music_dir = os.path.join('flask_server/preview_sound', instrument_dir)
    files = [f for f in os.listdir(music_dir) if f.endswith('.ogg') or f.endswith('.wav')]
    return jsonify(files)

@instrument_blueprint.route('/music/<instrument_code>/<filename>')
def serve_music_file(instrument_code, filename):
    instrument_dir = instrument.get(instrument_code)
    if not instrument_dir:
        return '', 404
    music_dir = os.path.join('preview_sound', instrument_dir)
    return send_from_directory(music_dir, filename)
