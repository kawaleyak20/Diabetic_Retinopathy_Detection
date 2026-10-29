"""Minimal local web interface for the DR mini-project."""

from pathlib import Path
import sys

from flask import Flask, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interface.predict import predict_scan

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB upload limit
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    patient_id = ""

    if request.method == "POST":
        patient_id = request.form.get("patient_id", "").strip()
        scan = request.files.get("scan_image")

        if not patient_id:
            error = "Enter a patient ID."
        elif not scan or not scan.filename:
            error = "Choose a retinal scan image."
        elif not allowed_file(scan.filename):
            error = "Upload a JPG, JPEG, PNG or WEBP image."
        else:
            try:
                result = predict_scan(scan.read())
                result["patient_id"] = patient_id
            except FileNotFoundError as exc:
                error = str(exc)
            except ValueError as exc:
                error = str(exc)
            except Exception as exc:
                app.logger.exception("Prediction failed")
                error = "Prediction failed. Check the active model and environment, then try again."

    return render_template("index.html", result=result, error=error, patient_id=patient_id)


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        result=None,
        error="Image is too large. Please upload an image smaller than 12 MB.",
        patient_id="",
    ), 413


if __name__ == "__main__":
    # Local-only development server; do not expose it to the public internet.
    app.run(host="127.0.0.1", port=5000, debug=False)
