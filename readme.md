##  Setup and Execution

Follow these steps to get the application running on your local machine or in an environment like Google Colab.

### 1. Prerequisites (Installation)

**Action:** Run this command in your terminal or Colab notebook:

```bash
pip install ultralytics gradio 
````

### 2. Prepare the Files

The application needs the Python script (app.py) and your model weights (yolov8s_thermal_prototype.pt) in the same directory.

#### A. Save the Python Script (app.py)

Save the full Gradio application code into a file named `app.py` in your project directory.

#### B. Provide the Model File

Place your trained YOLOv8 model file named `yolov8s_thermal_prototype.pt` in the same directory as `app.py`.

### 3. Run the Application

Navigate to your project directory and run:

```bash
python app.py
```