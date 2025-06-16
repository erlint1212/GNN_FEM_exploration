# utils.py
import sys
import os

class Tee(object):
    """
    A class to redirect stdout and stderr to both the console and a file.
    Acts as a context manager.
    """
    def __init__(self, filename, mode='a'):
        self.file = open(filename, mode, encoding='utf-8')
        self.stdout_original = sys.stdout
        self.stderr_original = sys.stderr
        # These are redirected when __enter__ is called by 'with' statement

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout_original # Restore original stdout
        sys.stderr = self.stderr_original # Restore original stderr
        if self.file:
            self.file.close()

    def write(self, data):
        # Write to the original console stdout
        self.stdout_original.write(data)
        self.stdout_original.flush()
        # Write to the log file
        if self.file and not self.file.closed:
            self.file.write(data)
            self.file.flush()

    def flush(self):
        self.stdout_original.flush()
        if self.file and not self.file.closed:
            self.file.flush()
        # Stderr might not always have flush or might be the same as stdout
        if hasattr(self.stderr_original, 'flush') and self.stderr_original is not self.stdout_original:
             self.stderr_original.flush()
