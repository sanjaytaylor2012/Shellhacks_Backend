import base64
import datetime
import sys

from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from flask_cors import CORS

import shellhacks_algorithm


UPLOAD_FOLDER = "static/uploads/"

app = Flask(__name__)
CORS(app)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 3 * 300 * 1024 * 1024


choice_to_video = {
    "Baseball": "../data/baseball/baseball_side_view.mp4",
    "Basketball": "",
    "Boxing": "",
    "Football": "",
    "Golf Side View": "",
    "Golf Back View": "",
    "Running": "",
    "Push-ups": "",
    "Snatch": "",
    "Clean and Jerk": ""
}


graph_file_names = [
    "Left arm angle.png",
    "Left arm bend angle.png",
    "Left leg bend angle.png",
    "Left waist bend angle.png",
    "Right arm angle.png",
    "Right arm bend angle.png",
    "Right leg bend angle.png",
    "Right waist bend angle.png"
]


@app.route('/upload', methods=['POST'])
def upload_video():
    r = request.json

    sport = r['currentSport']
    video = r['selectedFile']

    print("sport:", sport)

    video_bytes = base64.b64decode(video.split(',')[1])

    filename = sport + '-' + str(datetime.datetime.now().timestamp())
    filepath = UPLOAD_FOLDER + filename + '.mp4'

    with open(filepath, 'wb') as f:
        f.write(video_bytes)

    scores = shellhacks_algorithm.process_files(choice_to_video[sport], filepath)

    response = {}

    for graph in graph_file_names:
        with open(graph, "rb") as f:
            cur_graph_bytes = f.read()
            base_64_encoded_image = f'data:image/png;base64,{base64.b64encode(cur_graph_bytes)}'
            response[graph.split('.')[0]] = base_64_encoded_image

    return jsonify(response), 200


if __name__ == "__main__":
    app.run()

