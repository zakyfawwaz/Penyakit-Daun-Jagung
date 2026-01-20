# Setup MySQL untuk Aplikasi Corn Disease Detection

## 1. Install MySQL

### Windows:
1. Download MySQL dari https://dev.mysql.com/downloads/installer/
2. Install MySQL Server
3. Set password untuk root user saat instalasi

### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install mysql-server
sudo mysql_secure_installation
```

### macOS:
```bash
brew install mysql
brew services start mysql
```

## 2. Buat Database

Login ke MySQL:
```bash
mysql -u root -p
```

Buat database dan user:
```sql
CREATE DATABASE corn_disease_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER 'corn_user'@'localhost' IDENTIFIED BY 'your_password_here';
GRANT ALL PRIVILEGES ON corn_disease_db.* TO 'corn_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

## 3. Install Dependencies Python

```bash
pip install -r requirements.txt
```

Atau install PyMySQL secara manual:
```bash
pip install PyMySQL
```

## 4. Konfigurasi Aplikasi

### Opsi 1: Menggunakan Environment Variables (Recommended)

Buat file `.env` di root project:
```env
SECRET_KEY=your-secret-key-here
DB_USER=corn_user
DB_PASSWORD=your_password_here
DB_HOST=localhost
DB_PORT=3306
DB_NAME=corn_disease_db
```

Atau set langsung di sistem:
```bash
# Windows (PowerShell)
$env:DB_USER="corn_user"
$env:DB_PASSWORD="your_password"
$env:DB_HOST="localhost"
$env:DB_NAME="corn_disease_db"

# Linux/macOS
export DB_USER="corn_user"
export DB_PASSWORD="your_password"
export DB_HOST="localhost"
export DB_NAME="corn_disease_db"
```

### Opsi 2: Edit Langsung di app.py

Edit baris 19-26 di `app.py`:
```python
DB_USER = 'corn_user'
DB_PASSWORD = 'your_password_here'
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'corn_disease_db'
```

## 5. Jalankan Aplikasi

```bash
python app.py
```

Aplikasi akan otomatis membuat semua tabel yang diperlukan saat pertama kali dijalankan.

## 6. Migrasi Data dari SQLite (Opsional)

Jika Anda sudah punya data di SQLite dan ingin migrasi ke MySQL:

### Menggunakan SQLAlchemy:
```python
# Script migrasi sederhana
from app import app, db, User, DetectionHistory, DiseaseInfo
import sqlite3

# Connect ke SQLite lama
sqlite_conn = sqlite3.connect('instance/users.db')
sqlite_cursor = sqlite_conn.cursor()

# Ambil data users
users = sqlite_cursor.execute('SELECT * FROM user').fetchall()

# Insert ke MySQL
with app.app_context():
    for user in users:
        new_user = User(
            username=user[1],
            email=user[2],
            password_hash=user[3],
            full_name=user[4],
            phone=user[5] if len(user) > 5 else None,
            role=user[6] if len(user) > 6 else 'user'
        )
        db.session.add(new_user)
    db.session.commit()
```

## 7. Troubleshooting

### Error: "Access denied for user"
- Pastikan username dan password benar
- Pastikan user memiliki privileges untuk database

### Error: "Unknown database"
- Pastikan database sudah dibuat
- Cek nama database di konfigurasi

### Error: "Can't connect to MySQL server"
- Pastikan MySQL service berjalan
- Cek host dan port (default: localhost:3306)
- Cek firewall settings

### Error: "ModuleNotFoundError: No module named 'pymysql'"
```bash
pip install PyMySQL
```

## 8. Production Setup

Untuk production, gunakan environment variables dan pastikan:
- SECRET_KEY diubah menjadi random string yang kuat
- Database password kuat dan aman
- Backup database secara berkala
- Gunakan connection pooling (sudah dikonfigurasi di app.py)

## 9. Backup Database

```bash
# Backup
mysqldump -u corn_user -p corn_disease_db > backup.sql

# Restore
mysql -u corn_user -p corn_disease_db < backup.sql
```

