import os

class Misc:
    @staticmethod
    def create_directory(directory_path):
        """
        Creates a directory if it does not already exist.
        
        Parameters:
        directory_path (str): The path of the directory to create.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
        else:
            print(f"Directory '{directory_path}' already exists.")
