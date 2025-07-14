import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_env_int(key, default):
    """Safely get an integer from environment variables."""
    value = os.getenv(key, default)
    try:
        return int(str(value).split('#')[0].strip())
    except (ValueError, TypeError):
        return default

def get_env_float(key, default):
    """Safely get a float from environment variables."""
    value = os.getenv(key, default)
    try:
        return float(str(value).split('#')[0].strip())
    except (ValueError, TypeError):
        return default

class Config:
    # Flask Configuration
    SECRET_KEY = os.urandom(24)
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Email Configuration
    SMTP_SERVER = os.getenv('SMTP_SERVER')
    SMTP_PORT = get_env_int('SMTP_PORT', 587)
    SMTP_USERNAME = os.getenv('SMTP_USERNAME')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
    ALERT_EMAIL_FROM = os.getenv('ALERT_EMAIL_FROM')
    ALERT_EMAIL_TO = os.getenv('ALERT_EMAIL_TO')
    
    # Monitoring Configuration
    CRITICAL_CPU_THRESHOLD = get_env_float('CRITICAL_CPU_THRESHOLD', 80)
    CRITICAL_MEMORY_THRESHOLD = get_env_float('CRITICAL_MEMORY_THRESHOLD', 80)
    WARNING_CPU_THRESHOLD = get_env_float('WARNING_CPU_THRESHOLD', 60)
    WARNING_MEMORY_THRESHOLD = get_env_float('WARNING_MEMORY_THRESHOLD', 60)
    MONITORING_INTERVAL = get_env_int('MONITORING_INTERVAL', 300)
    
    # Cost Analysis Configuration
    COST_ALERT_THRESHOLD = get_env_float('COST_ALERT_THRESHOLD', 1000)
    COST_VARIANCE_THRESHOLD = get_env_float('COST_VARIANCE_THRESHOLD', 20)
    
    # API Keys
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
    GITLAB_TOKEN = os.getenv('GITLAB_TOKEN')
    
    # Celery Configuration
    CELERY_BROKER_URL = 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
    
    # File Export Configuration
    EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exports')
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # Logging Configuration
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Cache Configuration
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = 'redis://localhost:6379/1'
    CACHE_DEFAULT_TIMEOUT = 300 