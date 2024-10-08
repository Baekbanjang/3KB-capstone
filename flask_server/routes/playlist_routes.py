from flask import Blueprint, render_template, request, redirect, url_for, Response, jsonify
import bson
from bson import ObjectId
from utils.global_vars import fs, fs_files_collection, get_sorted_videos

playlist_blueprint = Blueprint('playlist', __name__)

@playlist_blueprint.route('/Playlist')
def playlist():
    # MongoDB에서 영상 리스트 가져오기
    sort_by = 'creationDate'
    sort_direction = 'asc'
    videolist = get_sorted_videos(sort_by, sort_direction)
    return render_template('Playlist.html', videos=videolist, sort_by=sort_by, sort_direction=sort_direction)  # MongoDB에서 모든 파일 조회

@playlist_blueprint.route('/Playlist/view/<video_id>', methods=['GET'])
def view(video_id):
    video = fs.find_one({'_id': bson.ObjectId(video_id)})
    def generate():
        for chunk in video:
            yield chunk
    if video:
        return Response(generate(), mimetype='video/mp4')
    else:
        return 'File not found', 404

@playlist_blueprint.route('/Playlist/rename/<video_id>', methods=['POST'])
def rename(video_id):
    new_name = request.form.get('new_name')
    result = fs_files_collection.update_one(
        {'_id': ObjectId(video_id)},
        {'$set': {'metadata.name': new_name}}
    )
    if result.matched_count == 0:
        return jsonify({'success': False, 'message': 'File not found'}), 404         
    return jsonify({'success': True, 'message': 'Video name updated successfully.'}), 200

@playlist_blueprint.route('/Playlist/delete_selected', methods=['POST'])
def delete_selected():
    video_ids_raw = request.form.getlist('video_ids')
    sort_by = request.form.get('sort_by', 'creationDate')
    sort_direction = request.form.get('sort_direction', 'asc')

    video_ids = []
    for video_id in video_ids_raw:
        video_ids.extend(video_id.split(','))
    
    for video_id in video_ids:
        fs.delete(ObjectId(video_id))
    return redirect(url_for('playlist.list_videos', sort_by=sort_by, sort_direction=sort_direction))

@playlist_blueprint.route('/Playlist/<sort_by>/<sort_direction>')
def list_videos(sort_by, sort_direction):
    videolist = get_sorted_videos(sort_by, sort_direction)
    # HTML 템플릿에 비디오 목록 전달
    return render_template('Playlist.html', videos=videolist, sort_by=sort_by, sort_direction=sort_direction)