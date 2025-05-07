import os

def create_directory_structure():
    # Define the directory structure
    directories = [
        'src/analysis',
        'src/models',
        'src/monitoring',
        'src/utils',
        'models/checkpoints',
        'data',
        'logs'
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    create_directory_structure() 