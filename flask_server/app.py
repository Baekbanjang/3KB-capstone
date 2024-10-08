from flask import Flask, render_template
from routes.playlist_routes import playlist_blueprint
from routes.movement_routes import movement_blueprint
from routes.instrument_routes import instrument_blueprint
from utils.global_vars import get_global_var

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# 블루프린트 등록
app.register_blueprint(playlist_blueprint)
app.register_blueprint(movement_blueprint)
app.register_blueprint(instrument_blueprint)

@app.route('/')
def home():
    gesture_preset = get_global_var('gesture_preset')
    pose_preset = get_global_var('pose_preset')
    return render_template('MovementPlay.html', gesture_preset=gesture_preset, pose_preset=pose_preset)

if __name__ == '__main__':
    app.run(threaded=True, debug=True)