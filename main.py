from roboflow import Roboflow
from dotenv import load_dotenv
import os


load_dotenv()

api_key = os.getenv('ROBOFLOW_PRIVATE_API')
project_id = os.getenv('PROJECT_ID')
# print(api_key)

rf = Roboflow(api_key=api_key)
project = rf.workspace().project(project_id)
print(project)
model = project.version(4).model
print(model)

model.predict("/Users/amit/Desktop/AadharCode/cv/barRoboFlow/images/ss1.png", confidence=20, overlap=10).save("/Users/amit/Desktop/AadharCode/cv/barRoboFlow/newImages/prediction.jpg")

# infer on a local image
print(model.predict("/Users/amit/Desktop/AadharCode/cv/barRoboFlow/images/ss1.png", confidence=40, overlap=30).json())