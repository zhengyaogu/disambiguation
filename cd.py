import os

class Cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.join(os.getcwd(), newPath)

    def __enter__(self):
        if not os.path.exists(self.newPath):
            os.mkdir(self.newPath)
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)