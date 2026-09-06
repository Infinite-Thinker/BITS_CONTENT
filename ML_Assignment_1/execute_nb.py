import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

print("Executing notebook headlessly to embed outputs and plots...")
notebook_path = "C:\\BITS_WILP\\ML_Assignment_1\\ML_Assignment1_AIML_2026.ipynb"

try:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    # We execute in the directory of the notebook so it can read/write relative files
    ep.preprocess(nb, {'metadata': {'path': 'C:\\BITS_WILP\\ML_Assignment_1'}})

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print("Notebook successfully executed and updated with inline outputs and plots!")
except Exception as e:
    print(f"Error during notebook execution: {e}")
