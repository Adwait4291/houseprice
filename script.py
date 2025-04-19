import os
import sys

def create_project_structure(project_name="ml_project"):
    """
    Creates a machine learning project structure.

    Args:
        project_name (str): The name of the project.  Defaults to "ml_project".
    """
    try:
        # Create the main project directory
        os.makedirs(project_name, exist_ok=True)
        print(f"Created project directory: {project_name}")

        # Define directories
        directories = [
            "data",
            "saved_model",
        ]

        # Create the directories
        for directory in directories:
            path = os.path.join(project_name, directory)
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

        # Define files and their content (empty for now, but you can add content later)
        files = {
            "data": ["AmesHousing.csv"],
            "saved_model": ["price_pipeline.joblib"],
            "": ["app.py", "api.py", "pipeline.py", "train.py", "requirements.txt", "README.md"],
        }

        # Create the files
        for directory, file_list in files.items():
            if directory:
                dir_path = os.path.join(project_name, directory)
            else:
                dir_path = project_name  # Root directory

            for file_name in file_list:
                file_path = os.path.join(dir_path, file_name)
                # Create empty file.
                open(file_path, 'a').close()
                print(f"Created file: {file_path}")

        print("Project structure created successfully.")

    except Exception as e:
        print(f"Error creating project structure: {e}")
        sys.exit(1)



if __name__ == "__main__":
    # Get project name from user
    project_name = input("Enter the project name (default: ml_project): ")
    if not project_name:
        project_name = "ml_project"  # Use default if user enters nothing

    create_project_structure(project_name)
