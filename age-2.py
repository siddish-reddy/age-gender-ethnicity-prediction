from io import BytesIO
from flask import Flask, make_response, request, render_template, send_file
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from pathlib import Path
import types
import os
from base64 import b64decode, b64encode
import time
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, extract_face
import tensorflow as tf

from tensorflow.keras.models import load_model

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.preprocessing import image

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

age_classes = ['Age_below20', 'Age_20_30','Age_30_40', 'Age_40_50', 'Age_above_50']
# emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
expression_objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
ethnicity_classes = ['white','indian',  'black', 'asian','arab', 'hispanic']
race_classes = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'hispanic'}
gender_classes = {0: 'male', 1: 'female'}
gender_map = dict((g, i) for i, g in gender_classes.items())
race_map = dict((r, i) for i, r in race_classes.items())

def origAgeLabel(age):
    if age<20:
      return 'Age_below20'
    if age<30:
      return 'Age_20_30'
    if age<40:
      return 'Age_30_40'
    if age<50:
      return 'Age_40_50'
    else:
      return 'Age_above50'

age_model = torchvision.models.resnet50()
#ethnicity_model = torchvision.models.resnet50()
#emotion_model = torchvision.models.resnet50()
face_detector = MTCNN(image_size=224, keep_all=True)

age_model.fc = torch.nn.Sequential(
    torch.nn.Linear(
        in_features = 2048,
        out_features = 5
    ),
    torch.nn.LogSoftmax(dim=1)
)
# ethnicity_model.fc = torch.nn.Sequential(
#     torch.nn.Linear(
#         in_features = 2048,
#         out_features = 6
#     ),
#     torch.nn.LogSoftmax(dim=1)
# )

# emotion_model.fc = torch.nn.Sequential(
#     torch.nn.Linear(
#         in_features = 2048,
#         out_features = 7
#     ),
#     torch.nn.LogSoftmax(dim=1)
# )

model = Sequential()

#1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
num_classes = 7
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy'
    , optimizer=tensorflow.keras.optimizers.Adam()
    , metrics=['accuracy']
)

age_model.load_state_dict(torch.load('models/age.pth'))
# ethnicity_model.load_state_dict(torch.load('models/ethnicity_model_temp-re-re.pth'))
# emotion_model.load_state_dict(torch.load('models/emotion.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print('using', device)
age_model.to(device)
# ethnicity_model.to(device)
# emotion_model.to(device)

multi_model = load_model('models/multi_task_model.h5')
model.load_weights('models/expression.h5')
preprocess = transforms.Compose([   transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

def label_faces(path):
    print(path)
    img = Image.open(path)
    if path.split('.')[-1] in  ['PNG', 'png']:
        img = img.convert('RGB')
    
    boxes, _ = face_detector.detect(img)
    img_cpy = img.copy()
    draw = ImageDraw.Draw(img_cpy)
    for box in boxes:
        i+=1
        draw.rectangle(box.tolist())
        face = img.crop(box).resize((198,198))
        face2 = face.resize((48, 48)).convert('L')
        x = image.img_to_array(face2)
        x = np.expand_dims(x, axis = 0)
        x /= 255
        custom = model.predict(x)
        emotion_label = expression_objects[custom.argmax()]
        age_pred, race_pred, gender_pred = multi_model.predict(np.expand_dims((np.asarray(face).astype('float')/255.0), axis=0))
        age_pred = int(age_pred[0] * 116.0)
        print(i,gender_pred)
        print(race_pred)
        gender_label = gender_classes[gender_pred.argmax()]
        inp = preprocess(face).unsqueeze(0)
        inp = inp.to(device)
        age_label = age_classes[torch.argmax(torch.exp(age_model.forward(inp)))]
        #emotion_label = emotion_classes[torch.argmax(torch.exp(emotion_model.forward(inp)))]
        #preds = torch.exp(ethnicity_model.forward(inp))
        #print(preds)
        #ethnicity_label = ethnicity(race_classes[race_pred.argmax()], ethnicity_classes[torch.argmax(preds)])
        ethnicity_label = race_classes[race_pred.argmax()]
        #print(age_label, emotion_label, ethnicity_label )
        print( age_pred,gender_label, emotion_label,  ethnicity_label)
        draw.text(box.tolist()[:2],  str(i)+origAgeLabel(age_pred)+', '+ emotion_label+', '+ ethnicity_label)
    img_io = BytesIO()
    img_cpy.save(img_io, 'JPEG')
    img_io.seek(0)
    return img_io

def ethnicity(a,b):
    print(a,b)
    if a==b:
        return a
    if a!='asian':
        return a
    if a =='asian' and b =='indian':
        return 'black'
    if a =='indian' and b =='asian':
        return 'indian'
    if a=='white' and b=='indian':
        return a
    if a!='indian' or b!='white':
        return b
    return a

#label_faces('tmp/test01.jpg')


app = Flask(__name__)
app.config['DEBUG'] = False
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG",]
app.config['UPLOAD_FOLDER'] = 'tmp/'

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route('/', methods=['GET'])
def index():
    return render_template('serving_template.html')

@app.route('/image', methods=["POST"])
def eval_image():
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request")
    if input_file.filename == '':
        return BadRequest("File name is not present in request")
    if not input_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return BadRequest("Invalid file type")
    
    filename = secure_filename(input_file.filename)
    input_file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    img_io = label_faces(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    # return send_file(b64encode(img_io.getvalue()), mimetype='image/jpeg',as_attachment=False)
    response = make_response(b64encode(img_io.getvalue()))
    response.headers['Content-Transfer-Encoding']='base64'
    return response

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000)