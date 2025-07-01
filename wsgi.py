import os
import sys

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app as application

if __name__ == "__main__":
    application.run()