"""
Dashboard Application
Modular Flask app with Blueprint architecture
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

# Import blueprints
from api.prompts import prompts_bp
from api.multimodal import multimodal_bp
from api.multiturn import multiturn_bp
from api.ctf import ctf_bp
from api.security import security_bp
from api.system import system_bp
from api.stats import stats_bp

# Get dashboard and project directories
DASHBOARD_DIR = Path(__file__).parent
PROJECT_ROOT = DASHBOARD_DIR.parent

# Initialize Flask app
app = Flask(
    __name__,
    template_folder=str(DASHBOARD_DIR / 'templates'),
    static_folder=str(DASHBOARD_DIR / 'static')
)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'prompt-arsenal-dashboard-secret-key-2025'
app.config['JSON_AS_ASCII'] = False  # Support Korean characters

# Register blueprints
app.register_blueprint(prompts_bp, url_prefix='/api/prompts')
app.register_blueprint(multimodal_bp, url_prefix='/api/multimodal')
app.register_blueprint(multiturn_bp, url_prefix='/api/multiturn')
app.register_blueprint(ctf_bp, url_prefix='/api/ctf')
app.register_blueprint(security_bp, url_prefix='/api/security')
app.register_blueprint(system_bp, url_prefix='/api/system')
app.register_blueprint(stats_bp, url_prefix='/api/stats')


# ============================================
# Routes
# ============================================

@app.route('/')
def index():
    """Home Dashboard"""
    return render_template('index.html')


@app.route('/prompts')
def prompts_page():
    """Text Prompts Page"""
    return render_template('modules/prompts.html')


@app.route('/multimodal')
def multimodal_page():
    """Multimodal Attacks Page"""
    return render_template('modules/multimodal.html')


@app.route('/multiturn')
def multiturn_page():
    """Multi-turn Campaigns Page"""
    return render_template('modules/multiturn.html')


@app.route('/ctf')
def ctf_page():
    """CTF Framework Page"""
    return render_template('modules/ctf.html')


@app.route('/security')
def security_page():
    """Security Scanner Page"""
    return render_template('modules/security.html')


@app.route('/system')
def system_page():
    """System Scanner Page"""
    return render_template('modules/system.html')


# ============================================
# Media File Serving
# ============================================

@app.route('/samples/<path:filename>')
def serve_samples(filename):
    """Serve files from samples directory"""
    return send_from_directory(str(PROJECT_ROOT / 'samples'), filename)


@app.route('/media/<path:filename>')
def serve_media(filename):
    """Serve files from media directory"""
    return send_from_directory(str(PROJECT_ROOT / 'media'), filename)


# ============================================
# Error Handlers
# ============================================

@app.errorhandler(404)
def not_found(error):
    """404 Error Handler"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """500 Error Handler"""
    return render_template('500.html'), 500


# ============================================
# Run Server
# ============================================

if __name__ == '__main__':
    print("""
╔═══════════════════════════════════════════════════════════╗
║      PROMPT ARSENAL - Dashboard Server v2.0              ║
╚═══════════════════════════════════════════════════════════╝

🚀 Server running on: http://localhost:5003
📊 Dashboard pages:
   - Home:        http://localhost:5003/
   - Prompts:     http://localhost:5003/prompts
   - Multimodal:  http://localhost:5003/multimodal
   - Multi-turn:  http://localhost:5003/multiturn
   - CTF:         http://localhost:5003/ctf
   - Security:    http://localhost:5003/security
   - System:      http://localhost:5003/system

Press CTRL+C to stop
    """)

    app.run(
        host='0.0.0.0',
        port=5003,
        debug=True,
        threaded=True
    )
