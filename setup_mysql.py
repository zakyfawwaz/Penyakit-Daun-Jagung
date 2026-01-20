"""
Script untuk setup MySQL database
Jalankan script ini setelah MySQL terinstall dan database dibuat
"""

import os
import sys

def check_mysql_connection():
    """Cek koneksi ke MySQL"""
    try:
        import pymysql
        
        # Konfigurasi default
        DB_USER = os.environ.get('DB_USER', 'root')
        DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
        DB_HOST = os.environ.get('DB_HOST', 'localhost')
        DB_PORT = int(os.environ.get('DB_PORT', '3306'))
        DB_NAME = os.environ.get('DB_NAME', 'corn_disease_db')
        
        print("Mencoba koneksi ke MySQL...")
        print(f"Host: {DB_HOST}:{DB_PORT}")
        print(f"Database: {DB_NAME}")
        print(f"User: {DB_USER}")
        
        # Coba koneksi
        connection = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8mb4'
        )
        
        print("✓ Koneksi berhasil!")
        connection.close()
        return True
        
    except ImportError:
        print("✗ PyMySQL belum terinstall. Jalankan: pip install PyMySQL")
        return False
    except pymysql.Error as e:
        print(f"✗ Error koneksi MySQL: {e}")
        print("\nPastikan:")
        print("1. MySQL sudah terinstall dan berjalan")
        print("2. Database sudah dibuat:")
        print(f"   CREATE DATABASE {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        print("3. User sudah dibuat dan memiliki privileges")
        print("4. Konfigurasi di app.py sudah benar")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("MySQL Connection Checker")
    print("=" * 50)
    
    if check_mysql_connection():
        print("\n✓ Semua siap! Anda bisa menjalankan: python app.py")
    else:
        print("\n✗ Perbaiki masalah di atas terlebih dahulu")
        sys.exit(1)

