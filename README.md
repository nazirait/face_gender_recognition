# Face & Gender Recognition.

The dataset is taken from Kaggle: https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

This repository contains the following files:
  1. main.py - code for face & gender recognition
  2. model_evaluation.py - evaluation of results
  3. face_detection.xml - trained model for face recognition
  4. gender_recognition.h5 - trained model for gender recognition
  5. Final_Report.pdf - the documentation for this project

The instruction on how to launch, configure, and use the solution is below:

1. Download the following files: main.py, evaluation.py, face_detection.xml and gender_recognition.h5. Make sure you are storing them in the same directory.

2. Install a compiler for Python (one of the following):
- PyCharm: https://www.jetbrains.com/pycharm/
- Anaconda: https://www.anaconda.com/products/distribution/

3. Installing the required packages in Python:
- For PyCharm: you need to go to PyCharm, temporarily create a project and open Terminal at the bottom
- For Anaconda: open Anaconda Prompt
In an open terminal or console, enter the following commands:
pip install cv2
pip install keras
pip install pandas

4. Open main.py:
- For PyCharm: File->Open, find and select main.py
- For Anaconda: Open Jupyter Notebook and File->Open, find and select main.py

5. 2 case scenarios:
- For real-time detection: Run the code and wait. The camera will appear and you will see the result.
- For detection by image: uncomment the lines [71, 83]. On line 71, provide path to the image as in example. Please, comment the code below line 85, i.e. [85, 112]. Run the code and wait. Once the execution is completed, you will get the result.

Insterested in model accuracy?

6. Open model_evaluation.ipynb. The instructions are in step 4 above.

7. Run the code and wait. Run the code and wait. Once the execution is completed, the output will be appeared. 
