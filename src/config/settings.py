# File: /Streamlit_app_deep/Streamlit_app_deep/src/config/settings.py

# Configuration settings for the Streamlit application

# General settings
PAGE_TITLE = "🧠 FlexiLearn AI"
PAGE_ICON = "🤖"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Model training settings
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32

# Data processing settings
IMAGE_SIZE = 64
TEST_SPLIT_RATIO = 0.2

# Allowed file types for uploads
ALLOWED_TABULAR_TYPES = ['csv', 'xlsx', 'txt']
ALLOWED_IMAGE_TYPES = ['zip', 'tgz']

# Other constants
RANDOM_STATE = 42