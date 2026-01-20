import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, inspect, cast, Date
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import io
import joblib
import numpy as np
from datetime import datetime, timedelta
import colorsys

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')

# Database configuration - SQLite
# Untuk menggunakan MySQL, set environment variable: USE_MYSQL=true
# dan set DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
USE_MYSQL = os.environ.get('USE_MYSQL', 'false').lower() == 'true'

if USE_MYSQL:
    # MySQL configuration (optional)
    try:
        DB_USER = os.environ.get('DB_USER', 'root')
        DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
        DB_HOST = os.environ.get('DB_HOST', 'localhost')
        DB_PORT = os.environ.get('DB_PORT', '3306')
        DB_NAME = os.environ.get('DB_NAME', 'corn_disease_db')
        
        # Test MySQL connection
        import pymysql
        test_conn = pymysql.connect(
            host=DB_HOST,
            port=int(DB_PORT),
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            connect_timeout=2
        )
        test_conn.close()
        
        # MySQL connection string
        app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 
            f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4')
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'pool_recycle': 300,
            'pool_pre_ping': True
        }
        print("✓ Using MySQL database")
    except Exception as e:
        print(f"⚠ MySQL connection failed: {e}")
        print("⚠ Falling back to SQLite database")
        USE_MYSQL = False

# Default: SQLite database
if not USE_MYSQL:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
    print("✓ Using SQLite database")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    role = db.Column(db.String(20), default='user')  # 'user' or 'admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        return self.role == 'admin'

# Models for Admin CRUD
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=True)
    author_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'), nullable=True)
    status = db.Column(db.String(20), default='pending')  # pending, published, draft
    thumbnail = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    author = db.relationship('User', backref='posts')
    category = db.relationship('Category', backref='posts')

class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    slug = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Media(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    uploader = db.relationship('User', backref='media')

class Page(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=True)
    slug = db.Column(db.String(200), unique=True, nullable=False)
    status = db.Column(db.String(20), default='draft')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    author_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, approved, rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    post = db.relationship('Post', backref='comments')
    author = db.relationship('User', backref='comments')

class DetectionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(500), nullable=False)
    cnn_prediction = db.Column(db.String(50), nullable=True)
    cnn_confidence = db.Column(db.Float, nullable=True)
    rf_prediction = db.Column(db.String(50), nullable=True)
    rf_confidence = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='detections')

class DiseaseInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    slug = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    symptoms = db.Column(db.Text, nullable=True)
    treatment = db.Column(db.Text, nullable=True)
    prevention = db.Column(db.Text, nullable=True)
    image_path = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Initialize database and create admin user
with app.app_context():
    # Check if role column exists in user table and add it if missing
    try:
        inspector = inspect(db.engine)
        if inspector.has_table('user'):
            columns = [col['name'] for col in inspector.get_columns('user')]
            if 'role' not in columns:
                # Add role column if it doesn't exist
                conn = db.engine.connect()
                conn.execute(db.text('ALTER TABLE user ADD COLUMN role VARCHAR(20) DEFAULT "user"'))
                conn.commit()
                conn.close()
                print("Added 'role' column to user table")
    except Exception as e:
        # If migration fails, try using raw SQL (SQLite compatible)
        print(f"Migration note: {e}")
        print("Attempting to add column with alternative method...")
        try:
            with db.engine.connect() as conn:
                conn.execute(db.text("ALTER TABLE user ADD COLUMN role VARCHAR(20) DEFAULT 'user'"))
                conn.commit()
            print("Added 'role' column using raw SQL")
        except Exception as e2:
            print(f"Could not add column: {e2}")
            print("Column may already exist or you may need to delete instance/users.db and restart")
    
    # Create all tables (this won't affect existing tables)
    db.create_all()
    
    # Create admin user if not exists
    try:
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@corn-disease.com',
                full_name='Administrator',
                role='admin'
            )
            admin.set_password('adminjagung123')
            db.session.add(admin)
            db.session.commit()
            print("Admin user created: username=admin, password=adminjagung123")
        else:
            # Update existing admin user to have admin role
            try:
                if not hasattr(admin, 'role') or (hasattr(admin, 'role') and admin.role != 'admin'):
                    admin.role = 'admin'
                    db.session.commit()
                    print("Updated existing admin user with admin role")
            except Exception as e:
                print(f"Note: Could not update admin role: {e}")
    except Exception as e:
        print(f"Note: Could not create/update admin user: {e}")

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model classes - URUTAN HARUS SAMA DENGAN ImageFolder (alfabetis: hawar, karat, sehat)
# ImageFolder mengurutkan kelas secara alfabetis, jadi urutannya: hawar (0), karat (1), sehat (2)
CLASS_NAMES = ['hawar', 'karat', 'sehat']

# ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load ResNet-50 model
def load_cnn_model():
    """Load the trained ResNet-50 model"""
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)  # 3 classes
    
    model_path = 'model/model_resnet50.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    else:
        return None

# Load Random Forest model
def load_rf_model():
    """Load the trained Random Forest model"""
    rf_model_path = 'model/model_random_forest.pkl'
    if os.path.exists(rf_model_path):
        return joblib.load(rf_model_path)
    else:
        return None

# Global model variables
cnn_model = load_cnn_model()
rf_model = load_rf_model()

def validate_image_is_leaf(image_path):
    """
    Validate if uploaded image is a leaf/plant using heuristic methods.
    Returns True if image appears to be a leaf, False otherwise.
    """
    try:
        # Open and convert image to RGB
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        
        # Resize if too large for faster processing
        if img_array.shape[0] > 1000 or img_array.shape[1] > 1000:
            image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
            img_array = np.array(image)
        
        # Convert RGB to HSV
        hsv_array = np.zeros_like(img_array, dtype=np.float32)
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                r, g, b = img_array[i, j] / 255.0
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                hsv_array[i, j] = [h * 360, s * 100, v * 100]  # H: 0-360, S: 0-100, V: 0-100
        
        # Extract HSV channels
        h_channel = hsv_array[:, :, 0]  # Hue: 0-360
        s_channel = hsv_array[:, :, 1]  # Saturation: 0-100
        v_channel = hsv_array[:, :, 2]  # Value: 0-100
        
        # Heuristic 1: Green color percentage
        # Green hue range: approximately 60-150 degrees (adjustable)
        green_mask = (h_channel >= 60) & (h_channel <= 150)
        # Also check saturation (leaves have good saturation) and value (not too dark)
        valid_green = green_mask & (s_channel >= 20) & (v_channel >= 20)
        green_percentage = np.sum(valid_green) / (img_array.shape[0] * img_array.shape[1]) * 100
        
        # Heuristic 2: Overall color distribution
        # Leaves should have some green, not just grayscale or other colors
        non_gray_mask = s_channel > 15  # Not grayscale
        color_diversity = np.sum(non_gray_mask) / (img_array.shape[0] * img_array.shape[1]) * 100
        
        # Heuristic 3: Brightness and contrast check
        # Leaves usually have reasonable brightness and contrast
        mean_brightness = np.mean(v_channel)
        std_brightness = np.std(v_channel)
        
        # Heuristic 4: Edge detection (simple gradient-based)
        # Convert to grayscale for edge detection
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        # Simple edge detection using Sobel-like gradient
        grad_x = np.abs(np.gradient(gray, axis=1))
        grad_y = np.abs(np.gradient(gray, axis=0))
        edge_strength = np.mean(grad_x + grad_y)
        
        # Decision logic
        is_leaf = False
        
        # Primary check: Green percentage (most important for leaves)
        if green_percentage >= 15:  # At least 15% green pixels
            is_leaf = True
        elif green_percentage >= 10 and color_diversity >= 30:  # Lower green but good color diversity
            is_leaf = True
        elif green_percentage >= 8 and mean_brightness >= 30 and edge_strength >= 10:
            # Very low green but reasonable brightness and texture
            is_leaf = True
        
        # Additional checks to reject obvious non-leaf images
        # Reject if too much is pure white/black (likely not a photo)
        white_black_percentage = np.sum((v_channel < 10) | (v_channel > 90)) / (img_array.shape[0] * img_array.shape[1]) * 100
        if white_black_percentage > 70:
            is_leaf = False
        
        # Reject if too uniform (likely a solid color or logo)
        if std_brightness < 5:
            is_leaf = False
        
        # Reject if too dark overall
        if mean_brightness < 15:
            is_leaf = False
        
        return is_leaf
        
    except Exception as e:
        print(f"Error in image validation: {str(e)}")
        # If validation fails, reject the image for safety
        return False

def extract_features(image_path, model):
    """Extract features from image using ResNet-50 (without FC layer)"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Create feature extractor (remove FC layer)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()
        model.eval()  # Ensure model is in eval mode
        
        with torch.no_grad():
            features = feature_extractor(image_tensor)
            features = features.view(features.size(0), -1)
            
        return features.cpu().numpy()
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def predict_cnn(image_path):
    """Predict using CNN ResNet-50"""
    if cnn_model is None:
        return None, 0.0
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = cnn_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
            predicted_class = CLASS_NAMES[predicted.item()]
            confidence_score = confidence.item() * 100
            
        return predicted_class, confidence_score
    except Exception as e:
        print(f"Error in CNN prediction: {str(e)}")
        return None, 0.0

def predict_rf(image_path):
    """Predict using Random Forest"""
    if rf_model is None or cnn_model is None:
        return None, 0.0
    
    try:
        # Extract features using CNN
        features = extract_features(image_path, cnn_model)
        if features is None:
            return None, 0.0
        
        # Predict using Random Forest
        prediction = rf_model.predict(features)[0]
        probabilities = rf_model.predict_proba(features)[0]
        confidence_score = np.max(probabilities) * 100
        
        predicted_class = CLASS_NAMES[prediction]
        
        return predicted_class, confidence_score
    except Exception as e:
        print(f"Error in RF prediction: {str(e)}")
        return None, 0.0

def predict_both_models(image_path):
    """Predict using both CNN and Random Forest models"""
    cnn_pred, cnn_conf = predict_cnn(image_path)
    rf_pred, rf_conf = predict_rf(image_path)
    
    return {
        'cnn': {'prediction': cnn_pred, 'confidence': cnn_conf},
        'rf': {'prediction': rf_pred, 'confidence': rf_conf}
    }

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Silakan login terlebih dahulu', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Silakan login terlebih dahulu', 'warning')
            return redirect(url_for('admin_login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin():
            flash('Akses ditolak. Hanya admin yang dapat mengakses halaman ini.', 'error')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Index page - redirect to login if not logged in"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    # Check if user is admin, redirect to admin dashboard
    user = User.query.get(session['user_id'])
    if user and user.is_admin():
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Username dan password harus diisi', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['full_name'] = user.full_name or user.username
            # Check if user is admin, redirect to admin dashboard
            if user.is_admin():
                session['is_admin'] = True
                flash(f'Selamat datang, {user.full_name or user.username}!', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash(f'Selamat datang, {user.full_name or user.username}!', 'success')
                return redirect(url_for('home'))
        else:
            flash('Username atau password salah', 'error')
    
    # If already logged in, redirect based on role
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.is_admin():
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('home'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register page"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        full_name = request.form.get('full_name')
        phone = request.form.get('phone')
        
        # Validation
        if not username or not email or not password:
            flash('Username, email, dan password harus diisi', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Password dan konfirmasi password tidak cocok', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username sudah digunakan', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email sudah terdaftar', 'error')
            return render_template('register.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            phone=phone
        )
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('Registrasi berhasil! Silakan login', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Terjadi kesalahan saat registrasi', 'error')
    
    # If already logged in, redirect to home
    if 'user_id' in session:
        return redirect(url_for('home'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout"""
    session.clear()
    flash('Anda telah logout', 'info')
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    """Home page"""
    user = User.query.get(session['user_id'])
    # Redirect admin to admin dashboard
    if user and user.is_admin():
        return redirect(url_for('admin_dashboard'))
    return render_template('home.html', user=user)

@app.route('/profile')
@login_required
def profile():
    """Profile page"""
    user = User.query.get(session['user_id'])
    # Redirect admin to admin dashboard
    if user and user.is_admin():
        return redirect(url_for('admin_dashboard'))
    return render_template('profile.html', user=user)

@app.route('/about')
@login_required
def about():
    """About page"""
    user = User.query.get(session['user_id'])
    # Redirect admin to admin dashboard
    if user and user.is_admin():
        return redirect(url_for('admin_dashboard'))
    return render_template('about.html', user=user)

@app.route('/technology')
@login_required
def technology():
    """Technology page"""
    user = User.query.get(session['user_id'])
    # Redirect admin to admin dashboard
    if user and user.is_admin():
        return redirect(url_for('admin_dashboard'))
    return render_template('technology.html', user=user)

@app.route('/detection')
@login_required
def detection():
    """Detection page - redirect to home with anchor"""
    user = User.query.get(session['user_id'])
    # Redirect admin to admin dashboard
    if user and user.is_admin():
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('home') + '#detection')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        flash('Tidak ada file yang diunggah', 'error')
        return redirect(url_for('detection'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('Tidak ada file yang dipilih', 'error')
        return redirect(url_for('detection'))
    
    if file:
        # Save uploaded file temporarily
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # VALIDASI: Check if image is a leaf BEFORE processing
        is_valid_leaf = validate_image_is_leaf(filepath)
        
        if not is_valid_leaf:
            # Delete the invalid image file
            try:
                os.remove(filepath)
            except:
                pass
            # Flash error message and redirect
            flash('Pastikan gambar yang Anda upload adalah gambar daun jagung.', 'error')
            return redirect(url_for('detection'))
        
        # If validation passes, proceed with prediction
        # Predict using both models
        results = predict_both_models(filepath)
        
        if results['cnn']['prediction'] is None and results['rf']['prediction'] is None:
            flash('Model tidak ditemukan. Silakan train model terlebih dahulu.', 'error')
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            return redirect(url_for('detection'))
        
        # Save relative path for template
        image_url = f"uploads/{filename}"
        
        # Set default values if model is not available
        cnn_pred = results['cnn']['prediction'] if results['cnn']['prediction'] else 'N/A'
        cnn_conf = results['cnn']['confidence'] if results['cnn']['prediction'] else 0.0
        rf_pred = results['rf']['prediction'] if results['rf']['prediction'] else 'N/A'
        rf_conf = results['rf']['confidence'] if results['rf']['prediction'] else 0.0
        
        user = User.query.get(session['user_id'])
        
        # Save detection history
        detection = DetectionHistory(
            user_id=user.id,
            image_path=image_url,
            cnn_prediction=cnn_pred if results['cnn']['prediction'] else None,
            cnn_confidence=cnn_conf if results['cnn']['prediction'] else None,
            rf_prediction=rf_pred if results['rf']['prediction'] else None,
            rf_confidence=rf_conf if results['rf']['prediction'] else None
        )
        db.session.add(detection)
        db.session.commit()
        
        return render_template('result.html', 
                             image_url=image_url,
                             cnn_prediction=cnn_pred,
                             cnn_confidence=cnn_conf,
                             rf_prediction=rf_pred,
                             rf_confidence=rf_conf,
                             cnn_available=results['cnn']['prediction'] is not None,
                             rf_available=results['rf']['prediction'] is not None,
                             user=user)
    
    return redirect(url_for('detection'))

# ==================== ADMIN ROUTES ====================

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Username dan password harus diisi', 'error')
            return render_template('admin/login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password) and user.is_admin():
            session['user_id'] = user.id
            session['username'] = user.username
            session['full_name'] = user.full_name or user.username
            session['is_admin'] = True
            flash(f'Selamat datang, {user.full_name or user.username}!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Username atau password salah, atau Anda bukan admin', 'error')
    
    # If already logged in as admin, redirect to dashboard
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.is_admin():
            return redirect(url_for('admin_dashboard'))
    
    return render_template('admin/login.html')

@app.route('/admin/logout')
def admin_logout():
    """Admin logout - redirect to regular login"""
    # Clear all session data
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('full_name', None)
    session.pop('is_admin', None)
    session.clear()
    flash('Anda telah logout', 'info')
    # Redirect to regular user login page
    return redirect('/login')

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard"""
    # Get statistics
    total_users = User.query.count()
    total_detections = DetectionHistory.query.count()
    total_diseases = DiseaseInfo.query.count()
    today_detections = DetectionHistory.query.filter(
        func.date(DetectionHistory.created_at) == func.date(datetime.utcnow())
    ).count()
    
    # Get recent detections
    recent_detections = DetectionHistory.query.order_by(DetectionHistory.created_at.desc()).limit(5).all()
    
    # Get detection statistics by disease type
    detection_stats = db.session.query(
        DetectionHistory.cnn_prediction,
        func.count(DetectionHistory.id).label('count')
    ).filter(DetectionHistory.cnn_prediction.isnot(None)).group_by(DetectionHistory.cnn_prediction).all()
    
    # Get top users by detection count
    top_users = db.session.query(
        User.username,
        func.count(DetectionHistory.id).label('detection_count')
    ).join(DetectionHistory, User.id == DetectionHistory.user_id).group_by(User.id).order_by(func.count(DetectionHistory.id).desc()).limit(5).all()
    
    return render_template('admin/dashboard.html',
                         total_users=total_users,
                         total_detections=total_detections,
                         total_diseases=total_diseases,
                         today_detections=today_detections,
                         recent_detections=recent_detections,
                         detection_stats=detection_stats,
                         top_users=top_users)

# ==================== ADMIN CRUD ROUTES ====================

# Posts CRUD
@app.route('/admin/posts')
@admin_required
def admin_posts():
    posts = Post.query.order_by(Post.created_at.desc()).all()
    return render_template('admin/posts/index.html', posts=posts)

@app.route('/admin/posts/create', methods=['GET', 'POST'])
@admin_required
def admin_posts_create():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        category_id = request.form.get('category_id') or None
        status = request.form.get('status', 'draft')
        
        post = Post(
            title=title,
            content=content,
            category_id=category_id,
            status=status,
            author_id=session['user_id']
        )
        db.session.add(post)
        db.session.commit()
        flash('Post berhasil dibuat', 'success')
        return redirect(url_for('admin_posts'))
    
    categories = Category.query.all()
    return render_template('admin/posts/create.html', categories=categories)

@app.route('/admin/posts/<int:id>/edit', methods=['GET', 'POST'])
@admin_required
def admin_posts_edit(id):
    post = Post.query.get_or_404(id)
    if request.method == 'POST':
        post.title = request.form.get('title')
        post.content = request.form.get('content')
        post.category_id = request.form.get('category_id') or None
        post.status = request.form.get('status', 'draft')
        post.updated_at = datetime.utcnow()
        db.session.commit()
        flash('Post berhasil diupdate', 'success')
        return redirect(url_for('admin_posts'))
    
    categories = Category.query.all()
    return render_template('admin/posts/edit.html', post=post, categories=categories)

@app.route('/admin/posts/<int:id>/delete', methods=['POST'])
@admin_required
def admin_posts_delete(id):
    post = Post.query.get_or_404(id)
    db.session.delete(post)
    db.session.commit()
    flash('Post berhasil dihapus', 'success')
    return redirect(url_for('admin_posts'))

# Categories CRUD
@app.route('/admin/categories')
@admin_required
def admin_categories():
    categories = Category.query.all()
    return render_template('admin/categories/index.html', categories=categories)

@app.route('/admin/categories/create', methods=['GET', 'POST'])
@admin_required
def admin_categories_create():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        slug = name.lower().replace(' ', '-')
        
        category = Category(name=name, slug=slug, description=description)
        db.session.add(category)
        db.session.commit()
        flash('Kategori berhasil dibuat', 'success')
        return redirect(url_for('admin_categories'))
    
    return render_template('admin/categories/create.html')

@app.route('/admin/categories/<int:id>/edit', methods=['GET', 'POST'])
@admin_required
def admin_categories_edit(id):
    category = Category.query.get_or_404(id)
    if request.method == 'POST':
        category.name = request.form.get('name')
        category.description = request.form.get('description')
        category.slug = category.name.lower().replace(' ', '-')
        db.session.commit()
        flash('Kategori berhasil diupdate', 'success')
        return redirect(url_for('admin_categories'))
    
    return render_template('admin/categories/edit.html', category=category)

@app.route('/admin/categories/<int:id>/delete', methods=['POST'])
@admin_required
def admin_categories_delete(id):
    category = Category.query.get_or_404(id)
    db.session.delete(category)
    db.session.commit()
    flash('Kategori berhasil dihapus', 'success')
    return redirect(url_for('admin_categories'))

# Users CRUD
@app.route('/admin/users')
@admin_required
def admin_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin/users/index.html', users=users)

@app.route('/admin/users/<int:id>/edit', methods=['GET', 'POST'])
@admin_required
def admin_users_edit(id):
    user = User.query.get_or_404(id)
    if request.method == 'POST':
        user.username = request.form.get('username')
        user.email = request.form.get('email')
        user.full_name = request.form.get('full_name')
        user.phone = request.form.get('phone')
        user.role = request.form.get('role', 'user')
        if request.form.get('password'):
            user.set_password(request.form.get('password'))
        db.session.commit()
        flash('User berhasil diupdate', 'success')
        return redirect(url_for('admin_users'))
    
    return render_template('admin/users/edit.html', user=user)

@app.route('/admin/users/<int:id>/delete', methods=['POST'])
@admin_required
def admin_users_delete(id):
    user = User.query.get_or_404(id)
    if user.is_admin() and User.query.filter_by(role='admin').count() == 1:
        flash('Tidak dapat menghapus admin terakhir', 'error')
        return redirect(url_for('admin_users'))
    db.session.delete(user)
    db.session.commit()
    flash('User berhasil dihapus', 'success')
    return redirect(url_for('admin_users'))

# Comments CRUD
@app.route('/admin/comments')
@admin_required
def admin_comments():
    comments = Comment.query.order_by(Comment.created_at.desc()).all()
    return render_template('admin/comments/index.html', comments=comments)

@app.route('/admin/comments/<int:id>/approve', methods=['POST'])
@admin_required
def admin_comments_approve(id):
    comment = Comment.query.get_or_404(id)
    comment.status = 'approved'
    db.session.commit()
    flash('Komentar berhasil disetujui', 'success')
    return redirect(url_for('admin_comments'))

@app.route('/admin/comments/<int:id>/reject', methods=['POST'])
@admin_required
def admin_comments_reject(id):
    comment = Comment.query.get_or_404(id)
    comment.status = 'rejected'
    db.session.commit()
    flash('Komentar berhasil ditolak', 'success')
    return redirect(url_for('admin_comments'))

@app.route('/admin/comments/<int:id>/delete', methods=['POST'])
@admin_required
def admin_comments_delete(id):
    comment = Comment.query.get_or_404(id)
    db.session.delete(comment)
    db.session.commit()
    flash('Komentar berhasil dihapus', 'success')
    return redirect(url_for('admin_comments'))

# ==================== DETECTION HISTORY ROUTES ====================

@app.route('/admin/detections')
@admin_required
def admin_detections():
    """Detection History page"""
    detections = DetectionHistory.query.order_by(DetectionHistory.created_at.desc()).all()
    return render_template('admin/detections/index.html', detections=detections)

@app.route('/admin/detections/<int:id>/delete', methods=['POST'])
@admin_required
def admin_detections_delete(id):
    detection = DetectionHistory.query.get_or_404(id)
    db.session.delete(detection)
    db.session.commit()
    flash('Riwayat deteksi berhasil dihapus', 'success')
    return redirect(url_for('admin_detections'))

# ==================== DISEASE INFO ROUTES ====================

@app.route('/admin/diseases')
@admin_required
def admin_diseases():
    """Disease Info page"""
    diseases = DiseaseInfo.query.order_by(DiseaseInfo.name).all()
    return render_template('admin/diseases/index.html', diseases=diseases)

@app.route('/admin/diseases/create', methods=['GET', 'POST'])
@admin_required
def admin_diseases_create():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        symptoms = request.form.get('symptoms')
        treatment = request.form.get('treatment')
        prevention = request.form.get('prevention')
        slug = name.lower().replace(' ', '-')
        
        disease = DiseaseInfo(
            name=name,
            slug=slug,
            description=description,
            symptoms=symptoms,
            treatment=treatment,
            prevention=prevention
        )
        db.session.add(disease)
        db.session.commit()
        flash('Informasi penyakit berhasil dibuat', 'success')
        return redirect(url_for('admin_diseases'))
    
    return render_template('admin/diseases/create.html')

@app.route('/admin/diseases/<int:id>/edit', methods=['GET', 'POST'])
@admin_required
def admin_diseases_edit(id):
    disease = DiseaseInfo.query.get_or_404(id)
    if request.method == 'POST':
        disease.name = request.form.get('name')
        disease.description = request.form.get('description')
        disease.symptoms = request.form.get('symptoms')
        disease.treatment = request.form.get('treatment')
        disease.prevention = request.form.get('prevention')
        disease.slug = disease.name.lower().replace(' ', '-')
        disease.updated_at = datetime.utcnow()
        db.session.commit()
        flash('Informasi penyakit berhasil diupdate', 'success')
        return redirect(url_for('admin_diseases'))
    
    return render_template('admin/diseases/edit.html', disease=disease)

@app.route('/admin/diseases/<int:id>/delete', methods=['POST'])
@admin_required
def admin_diseases_delete(id):
    disease = DiseaseInfo.query.get_or_404(id)
    db.session.delete(disease)
    db.session.commit()
    flash('Informasi penyakit berhasil dihapus', 'success')
    return redirect(url_for('admin_diseases'))

# ==================== STATISTICS ROUTES ====================

@app.route('/admin/statistics')
@admin_required
def admin_statistics():
    """Statistics page with detailed analytics"""
    # Overall statistics
    total_users = User.query.count()
    total_detections = DetectionHistory.query.count()
    total_diseases = DiseaseInfo.query.count()
    
    # Time-based statistics
    # Get current date - compatible with both MySQL and SQLite
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_detections = DetectionHistory.query.filter(
        DetectionHistory.created_at >= today_start
    ).count()
    
    # Calculate week and month ago as datetime objects
    week_ago = datetime.utcnow() - timedelta(days=7)
    month_ago = datetime.utcnow() - timedelta(days=30)
    
    # Date filtering - compatible with both MySQL and SQLite
    week_detections = DetectionHistory.query.filter(
        DetectionHistory.created_at >= week_ago
    ).count()
    
    month_detections = DetectionHistory.query.filter(
        DetectionHistory.created_at >= month_ago
    ).count()
    
    # Detection statistics by disease type
    detection_by_disease = db.session.query(
        DetectionHistory.cnn_prediction,
        func.count(DetectionHistory.id).label('count')
    ).filter(DetectionHistory.cnn_prediction.isnot(None)).group_by(DetectionHistory.cnn_prediction).all()
    
    # Detection statistics by date (last 7 days)
    # SQLite compatible (using strftime)
    week_ago_str = week_ago.strftime('%Y-%m-%d')
    daily_detections_raw = db.session.query(
        func.strftime('%Y-%m-%d', DetectionHistory.created_at).label('date'),
        func.count(DetectionHistory.id).label('count')
    ).filter(func.strftime('%Y-%m-%d', DetectionHistory.created_at) >= week_ago_str).group_by(func.strftime('%Y-%m-%d', DetectionHistory.created_at)).order_by(func.strftime('%Y-%m-%d', DetectionHistory.created_at)).all()
    
    # Convert to list of dicts with proper date formatting
    daily_detections = []
    for row in daily_detections_raw:
        if row.date:
            try:
                # Parse the date string and format it
                date_obj = datetime.strptime(row.date, '%Y-%m-%d')
                date_str = date_obj.strftime('%d %b')
            except Exception:
                date_str = str(row.date)
            daily_detections.append({'date': date_str, 'count': row.count})
        else:
            daily_detections.append({'date': 'N/A', 'count': row.count})
    
    # Top users by detection count
    top_users = db.session.query(
        User.username,
        User.full_name,
        func.count(DetectionHistory.id).label('detection_count')
    ).join(DetectionHistory, User.id == DetectionHistory.user_id).group_by(User.id).order_by(func.count(DetectionHistory.id).desc()).limit(10).all()
    
    # Average confidence by disease
    avg_confidence = db.session.query(
        DetectionHistory.cnn_prediction,
        func.avg(DetectionHistory.cnn_confidence).label('avg_conf')
    ).filter(DetectionHistory.cnn_prediction.isnot(None), DetectionHistory.cnn_confidence.isnot(None)).group_by(DetectionHistory.cnn_prediction).all()
    
    return render_template('admin/statistics.html',
                         total_users=total_users,
                         total_detections=total_detections,
                         total_diseases=total_diseases,
                         today_detections=today_detections,
                         week_detections=week_detections,
                         month_detections=month_detections,
                         detection_by_disease=detection_by_disease,
                         daily_detections=daily_detections,
                         top_users=top_users,
                         avg_confidence=avg_confidence)

# ==================== SETTINGS ROUTES ====================

@app.route('/admin/settings', methods=['GET', 'POST'])
@admin_required
def admin_settings():
    """Settings page"""
    if request.method == 'POST':
        # Handle settings update
        setting_type = request.form.get('setting_type')
        
        if setting_type == 'general':
            # General settings can be added here
            flash('Pengaturan berhasil disimpan', 'success')
        elif setting_type == 'model':
            # Model settings can be added here
            flash('Pengaturan model berhasil disimpan', 'success')
        
        return redirect(url_for('admin_settings'))
    
    # Get current statistics for display
    total_users = User.query.count()
    total_detections = DetectionHistory.query.count()
    
    return render_template('admin/settings.html',
                         total_users=total_users,
                         total_detections=total_detections)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

