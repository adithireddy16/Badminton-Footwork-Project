
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import subprocess

from process_video import process_video


# Main project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)


# Folders
UPLOAD_FOLDER = os.path.join(BASE_DIR, "backend", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "backend", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# FFmpeg location
FFMPEG = r"C:\Users\Aditi Reddy\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"


@app.route("/")
def home():
    return "Badminton Footwork Detection Backend is Running!"


@app.route("/upload", methods=["POST"])
def upload_video():

    if "video" not in request.files:
        return jsonify({
            "error": "No video uploaded"
        }), 400

    video = request.files["video"]

    if video.filename == "":
        return jsonify({
            "error": "No video selected"
        }), 400

    # Save uploaded video
    input_path = os.path.join(
        UPLOAD_FOLDER,
        video.filename
    )

    video.save(input_path)

    print("Video uploaded:", video.filename)

    # OpenCV output
    base_name = os.path.splitext(video.filename)[0]

    processed_filename = "processed_" + base_name + ".mp4"
    processed_path = os.path.join(
        RESULT_FOLDER,
        processed_filename
    )

    print("Starting YOLO video processing...")

    success = process_video(
        input_path,
        processed_path
    )

    if not success:
        return jsonify({
            "error": "Video processing failed"
        }), 500

    print("YOLO video processing completed!")

    # Browser-compatible H.264 output
    h264_filename = (
        "processed_" +
        base_name +
        "_h264.mp4"
    )

    h264_path = os.path.join(
        RESULT_FOLDER,
        h264_filename
    )

    print("Converting video to browser-compatible format...")

    try:

        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-i",
                processed_path,
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                h264_path
            ],
            check=True
        )

    except Exception as e:

        print("FFmpeg conversion failed:", e)

        return jsonify({
            "error": "Video conversion failed"
        }), 500

    print("Browser-compatible video created!")

    return jsonify({
        "message": "Video processed successfully!",
        "filename": video.filename,
        "output": h264_filename
    })


@app.route("/results/<filename>")
def get_result(filename):

    return send_from_directory(
        RESULT_FOLDER,
        filename
    )


if __name__ == "__main__":
    app.run(debug=True)

