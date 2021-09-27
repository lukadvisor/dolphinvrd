import streamlit as st
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import os

import cv2
import imageio

import numpy as np
import torch, torchvision

from model.helper.parser import DolphinParser
from model.helper.dolphin_detector_train import get_transform
from model.helper.utility import git_root, cpu_or_gpu, plot_traj
from dataset.dolphin import DOLPHIN, DOLPHINVIDEOVRD, TESTER

import os, yaml
from tqdm import tqdm

from model.tracker.tracktor.network import FRCNN_FPN
from model.tracker.tracktor.tracktor import Tracker

from model.motiondetect.s3d_resnet import s3d_resnet


##########
##### Set up sidebar.
##########


def main2():

    dp = DolphinParser()
    dp_args = dp.parse()
    DEVICE = cpu_or_gpu(dp_args.device)

    
    #torch.manual_seed(1234)
    #torch.cuda.manual_seed(1234)
    #np.random.seed(1234)
    #torch.backends.cudnn.deterministic = True


    print("+Initializing object detector+")

    try:
        obj_detect = FRCNN_FPN(num_classes=3)
        model_weight = os.path.join(git_root(), 'model', 'param', 'general_detector_30.pth') #'model', 'param', 'general_detector_0.pth')
        # .pth file needed

        checkpoint = torch.load(model_weight, map_location=DEVICE)
        obj_detect.load_state_dict(checkpoint['model_state_dict'])

    except (FileNotFoundError, Exception):
        print('Failed Loading Default Object Detector, Use torchvision instead')
        obj_detect = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    obj_detect.to(DEVICE)
    obj_detect.eval()

    print("+Initializing Tracker")
    tracker = None
    with open(os.path.join(git_root(), 'model', 'tracker', 'tracktor', 'configuration.yaml'), 'r') as stream:
        try:
            configyaml = yaml.safe_load(stream)['tracktor']['tracker']
            tracker = Tracker(obj_detect, None, configyaml, DEVICE)
        except yaml.YAMLError as exc:
            print(exc)


    dataset = TESTER(data_path='./temDir/',
                      set='Test',
                      mode='general',
                      transforms=get_transform(train=False))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


    tracker.reset()
    for i, (clip, blob) in enumerate(dataloader):
        with torch.no_grad():
            tracker.step(blob, idx=i+1)
        
    traj = tracker.get_results()

    print(traj)
    visualise = plot_traj_2(clip, traj, dataloader, None)



    gif_images = []
    for i, pic in enumerate(visualise):
        gif_images.append(pic[:, :, ::-1])
    imageio.mimsave('demo.gif', gif_images)



def plot_traj_2(clip, traj, dler, motion=None):
    

    clip_np  = [ cv2.imread(c['img_path'][0]) for _, c in dler]

    print(traj)
    for id, cood in traj.items():
        
        
        for f, cord in cood.items():
        
            bbox = cord[0:4]
            x_min, y_min, x_max, y_max = [ int(b.item()) for b in bbox]
            
            this_img = clip_np[f]
            this_img = cv2.rectangle(this_img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            centre_x = int(x_min/2 + x_max/2)
            centre_y = int(y_min/2 + y_max/2)
            this_img = cv2.putText(this_img, f'ID: {id}', (centre_x, centre_y), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            

            clip_np[f] = this_img

    return clip_np


# Add in location to select image.

## Add in sliders.
confidence_threshold = st.sidebar.slider('Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?', 0.0, 1.0, 0.5, 0.01)
overlap_threshold = st.sidebar.slider('Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?', 0.0, 1.0, 0.5, 0.01)



#image = Image.open('example/000001/000000.png')
#st.sidebar.image(image,
#                 use_column_width=True)

#image = Image.open('example/000001/000000.png')
#st.sidebar.image(image,
#                 use_column_width=True)

##########
##### Set up main app.
##########

## Title.
st.write('# Dolphin Detection')


st.write('#### Select image or video to upload.')
uploaded_file = st.file_uploader('', type=['png', 'jpg', 'jpeg', 'mp4', 'avi'])


## Pull in default image or user-selected image.
#uploaded_file = 'example/000000.gif'   # 1/000000.png'
#uploaded_file = 'example/000001/000000.png'   # 1/000000.png'

if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    # User-selected image.
	with open(os.path.join("temDir", 'target.mp4'), "wb") as f: 
	      f.write(uploaded_file.getbuffer())  

	

#### Run main script

if uploaded_file is not None:  
    print("+Sorting out the video+")
    cap= cv2.VideoCapture('./temDir/target.mp4')
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(f'./temDir/{str(i).zfill(6)}.jpg', frame)
        i+=1
    cap.release()
    os.system('rm ./temDir/target.mp4')


    print('!!!!RUNNING MAIN2!!!!')
    main2()



## Subtitle.
st.write('### Inferenced Image/Video')

# Convert to JPEG Buffer.
buffered = io.BytesIO()
#image.save(buffered, quality=90, format='JPEG')

# Base 64 encode.
img_str = base64.b64encode(buffered.getvalue())
img_str = img_str.decode('ascii')
## Construct the URL to retrieve image.

#image = Image.open(BytesIO(r.content))
# Convert to JPEG Buffer.
buffered = io.BytesIO()
#image.save(buffered, quality=90, format='JPEG')

# Display image.
#st.image(image, use_column_width=True)



file_ = open("./demo.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()




motionfile_ = open("example/motionplot.gif", "rb")
morioncontents = motionfile_.read()
motion_data_url = base64.b64encode(morioncontents).decode("utf-8")
motionfile_.close()


if uploaded_file:
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="alt gif">',
        unsafe_allow_html=True
    )

    ## Generate list of confidences.
    #onfidences = [box['confidence'] for box in output_dict['predictions']]
    confidences = [0.7, 0.6, 0.5]

    ## Summary statistics section in main app.
    st.write('### Summary Statistics')
    st.write(f'Number of Bounding Boxes (ignoring overlap thresholds): {len(confidences)}')
    st.write(f'Average Confidence Level of Bounding Boxes: {(np.round(np.mean(confidences),4))}')

    st.markdown(
        f'<img src="data:image/gif;base64,{motion_data_url}" alt="alt gif">',
        unsafe_allow_html=True
    )
    
    ## Histogram in main app.
    st.write('### Histogram of Confidence Levels')
    fig, ax = plt.subplots()
    ax.hist(confidences, bins=10, range=(0.0,1.0))
    st.pyplot(fig)

    agree = st.checkbox('Export JSON File')
    if agree:
        st.write('Success')


    #form = st.form(key='my-form')
    #submit = form.form_submit_button('Export JSON File')

    #if submit:
    #    st.write(f'Saved Json file')
