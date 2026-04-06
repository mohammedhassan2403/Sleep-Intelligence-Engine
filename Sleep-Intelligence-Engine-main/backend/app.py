from flask import Flask
from flask_cors import CORS
from routes.sleep_routes import sleep_bp

app = Flask(__name__)

# ✅ ENABLE CORS (VERY IMPORTANT)
CORS(app)

# Register Blueprint
app.register_blueprint(sleep_bp, url_prefix="/api")

# Health check route
@app.route("/")
def home():
    return {
        "message": "Sleep Intelligence Engine API is running 🚀",
        "endpoints": {
            "predict": "/api/predict (POST)"
        }
    }

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
