import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash, session
from werkzeug.utils import secure_filename
from supabase import create_client, Client
from dotenv import load_dotenv
import bcrypt
from ocr_utils import ocr_with_tesseract, ocr_with_easyocr
from ai_gloss import kb_constrained_sentence  # 👈 your gloss function

# -------------------- CONFIG --------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "tiff", "bmp"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
app.secret_key = "supersecretkey"  # change in production

# -------------------- SUPABASE --------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- HELPERS --------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def login_required(func):
    """Decorator to require login for OCR routes"""
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            flash("Please login first", "danger")
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return wrapper

# -------------------- AUTH ROUTES --------------------
@app.route("/", methods=["GET"])
def start():
    """Redirect starting page to signup"""
    return redirect(url_for("signup"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        try:
            supabase.table("users").insert({
                "username": username,
                "password_hash": hashed_pw
            }).execute()
            session["user"] = username
            flash("Signup successful!", "success")
            return redirect(url_for("index"))  # go to OCR page
        except Exception as e:
            flash(f"Error: {e}", "danger")

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        response = supabase.table("users").select("*").eq("username", username).execute()
        if response.data:
            user = response.data[0]
            if bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
                session["user"] = username
                flash("Login successful!", "success")
                return redirect(url_for("index"))  # go to OCR page
            else:
                flash("Invalid password!", "danger")
        else:
            flash("User not found!", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

# -------------------- OCR ROUTE --------------------
@app.route("/index", methods=["GET", "POST"])
@login_required
def index():
    text = None
    gloss = None
    img_url = None

    if request.method == "POST":
        if 'file' not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No selected file", "danger")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            backend = request.form.get("backend", "tesseract")

            try:
                if backend == "easyocr":
                    text = ocr_with_easyocr(path)
                else:
                    text = ocr_with_tesseract(path, lang='eng', psm=3, oem=3, resize_factor=1.3)

                if text:
                    gloss = kb_constrained_sentence(text)

            except Exception as e:
                text = f"OCR failed: {e}"
                flash(f"OCR failed: {e}", "danger")

            img_url = url_for('uploaded_file', filename=filename)

    return render_template("index.html", text=text, gloss=gloss, img_url=img_url)

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
