import base64
import datetime
import sys

from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from flask_cors import CORS

sys.path.append("../")

import shellhacks_algorithm


UPLOAD_FOLDER = "static/uploads/"

app = Flask(__name__)
CORS(app)
app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 3 * 300 * 1024 * 1024


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
    "Clean and Jerk": "",
}


@app.route("/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        flash("No file part")
        print("No file part")
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        flash("No image selected for uploading")
        print("No image selected for uploading")
        return redirect(request.url)
    else:
        filename = file.filename
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        flash("Video successfully uploaded and displayed below")
        print("Video successfully uploaded and displayed below")
        return redirect(request.url)


@app.route("/display/<filename>")
def display_video(filename):
    # print('display_video filename: ' + filename)
    return redirect(url_for("static", filename="uploads/" + filename), code=301)


if __name__ == "__main__":
    app.run()
