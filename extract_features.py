import os

import pandas as pd
import numpy as np
import torch
import cv2

import datetime as dt
from datetime import datetime

from models.pytorch_i3d import InceptionI3d
from cropper import crop_face

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_rgb_frames(image_dir, vid, start, num, desired_channel_order='rgb'):
    frames = []
    for i in range(start, start + num):
        img = cv2.imread(os.path.join(image_dir, vid, "images" + str(i).zfill(4) + '.png'))

        if desired_channel_order == 'bgr':
            img = img[:, :, [2, 1, 0]]

        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def crop_video(path, iloc):

    filename = path+str(iloc[0])+".mp4"
    init = iloc[1]
    end = iloc[2]
    
    frames = []
    
    cap = cv2.VideoCapture(filename) #Read Frame
    fps = cap.get(cv2.CAP_PROP_FPS) #Extract the frame per second (fps)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #width

    origin = "00:00:00.00" #the origin
    start = str(dt.timedelta(seconds=init)) #specify start time in hh:mm:ss
    end = str(dt.timedelta(seconds=end)) #specify end time in hh:mm:ss

    if not '.' in start:
        start += ".00"
        
    if not '.' in end:
        end += ".00"

    origintime = datetime.strptime(origin,'%H:%M:%S.%f') #origin 
    starttime = datetime.strptime(start,'%H:%M:%S.%f') #start time
    endtime = datetime.strptime(end,'%H:%M:%S.%f') #end time
    
    first = starttime-origintime
    last = endtime-origintime
    
    startframe = fps*(first).total_seconds()  #get the start frame
    endframe = fps*(last).total_seconds()  #get the end frame

    counter = 1 #set counter
    while(cap.isOpened()): #while the cap is open

        ret, frame = cap.read() #read frame
        if frame is None: #if frame is None
            break  
    
        frame = cv2.resize(frame, (width,height)) #resize the frame
        if counter >= startframe and counter <= endframe: #check for range of output
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        counter+=1  #increase counter
    
    return frames

def load_all_rgb_frames_from_how2sign(path, iloc, desired_channel_order='rgb'):
    frames = crop_video(path, iloc)
    array = []

    for i in range(len(frames)):
        frame = cv2.resize(frames[i], dsize=(224, 224))

        frame_transformed = frame.copy()   
            
        if desired_channel_order == 'bgr':
            frame_transformed = frame_transformed[:, :, [2, 1, 0]]

        frame_transformed = (frame_transformed / 255.) * 2 - 1
        array.append(frame_transformed)

    nframes = np.asarray(array, dtype=np.float32)
    return nframes




def load_all_rgb_frames_from_video(video, desired_channel_order='rgb'):
    cap = cv2.VideoCapture(video)
    
    frames = []
    faces = []
    
    #last_cropped = np.zeros((224,224,3), np.uint8)

    while(True):

        frame = np.zeros((224,224,3), np.uint8)

        try:
            ret, frame = cap.read()
            frame = cv2.resize(frame, dsize=(224, 224))

            frame_transformed = frame.copy()   
            
            if desired_channel_order == 'bgr':
                frame_transformed = frame_transformed[:, :, [2, 1, 0]]

            frame_transformed = (frame_transformed / 255.) * 2 - 1
            frames.append(frame_transformed)

        except:
            break


        #Face Extractor
        cropped = crop_face(frame.copy())
        try:
            cropped = cv2.resize(cropped, dsize=(224, 224))
            #last_cropped = cropped.copy()
        except:
            cropped = np.zeros((224,224,3), np.uint8)
            print("catch resize!")
        cropped = (cropped / 255.) * 2 - 1
        faces.append(cropped)


    nframes = np.asarray(frames, dtype=np.float32)
    nfaces = np.asarray(faces, dtype=np.float32)
    
    return nframes, nfaces

'''
def load_rgb_frames_from_video(vid_path, vid, start, num, resize=(256, 256), desired_channel_order='rgb'):
    vidcap = cv2.VideoCapture(video_path)

    frames = []

    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(min(num, int(total_frames - start))):
        success, img = vidcap.read()

        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > 256 or h > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

        img = (img / 255.) * 2 - 1

        if desired_channel_order == 'bgr':
            img = img[:, :, [2, 1, 0]]

        frames.append(img)

    return np.asarray(frames, dtype=np.float32)
'''

def extract_features_fullvideo(model, inp, framespan, stride):
    rv = []

    indices = list(range(len(inp)))
    groups = []
    for ind in indices:

        if ind % stride == 0:
            groups.append(list(range(ind, ind+framespan)))

    for g in groups:
        # numpy array indexing will deal out-of-index case and return only till last available element
        frames = inp[g[0]: min(g[-1]+1, inp.shape[0])]

        num_pad = 9 - len(frames)
        if num_pad > 0:
            pad = np.tile(np.expand_dims(frames[-1], axis=0), (num_pad, 1, 1, 1))
            frames = np.concatenate([frames, pad], axis=0)

        frames = frames.transpose([3, 0, 1, 2])

        ft = _extract_features(model, frames)

        rv.append(ft)

    return rv


def _extract_features(model, frames):
    inputs = torch.from_numpy(frames)

    inputs = inputs.unsqueeze(0)

    inputs = inputs.cuda()
    with torch.no_grad():
        ft = model.extract_features(inputs)
    ft = ft.squeeze(-1).squeeze(-1)[0].transpose(0, 1)

    print("OUTSIDE", ft.size())

    ft = ft.cpu()

    return ft


def run(weight, frame_roots, outroot, inp_channels='rgb'):
    videos = []

    for root in frame_roots:
        paths = sorted(os.listdir(root))
        videos.extend([os.path.join(root, path) for path in paths])

    # ===== setup models ======
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(2000)
    i3d.load_state_dict(torch.load(weight)) # Network's Weight
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode
    

    # Face model feature extractor
    #fmodel = InceptionI3d(400, in_channels=3)
    #fmodel.replace_logits(2000)
    #fmodel.cuda()
    #fmodel.train(False) # Set model to evaluate mode


    print('feature extraction starts.')

    # ===== extract features ======
    for framespan, stride in [(16, 2), (12, 2), (8, 2)]:

        outdir = os.path.join(outroot, 'span={}_stride={}'.format(framespan, stride))

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for ind, video in enumerate(videos):
            out_path = os.path.join(outdir, os.path.basename(video[:-4])) + '.pt'

            #with open('./done.txt') as file:
            #    if out_path in file.read():
            #        print('{} exists, continue'.format(out_path))
            #        continue

            #if os.path.exists(out_path):
            #    print('{} exists, continue'.format(out_path))
            #    continue

            frames, face_frames = load_all_rgb_frames_from_video(video, inp_channels)
            
            features = extract_features_fullvideo(i3d, frames, framespan, stride)
            #face_features = extract_features_fullvideo(fmodel, face_frames, framespan, stride)

            #CONCATENADO
            #for i in range(len(face_features)):
            #    features.append(face_features[i])

            print(ind, video, len(features))

            torch.save(features, os.path.join(outdir, os.path.basename(video[:-4])) + '.pt')

            #with open('./done.txt', 'a') as f:
            #    f.write(out_path + "\n")
            #    f.close()



def run_h2s(weight, path_data, videos_folder, outroot, inp_channels='rgb'):

    text_file = open(path_data, "r")

    lines = text_file.readlines()
    video = list(map(lambda elem: str(elem.split('\t')[0]), lines))
    init = list(map(lambda elem: float(elem.split('\t')[1]), lines))
    end = list(map(lambda elem: float(elem.split('\t')[2]), lines))
    frase = list(map(lambda elem: str(elem.split('\t')[3]).replace('\n',''), lines))

    df = pd.DataFrame({'video': video, 'init': init, 'end': end, 'frase': frase})

    # ===== setup models ======
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(2000)
    i3d.load_state_dict(torch.load(weight))
    i3d.cuda()
    i3d.train(False)

    print('feature extraction starts.')

    # ===== extract features ======
    for framespan, stride in [(16, 2), (12, 2), (8, 2)]:

        outdir = os.path.join(outroot, 'span={}_stride={}'.format(framespan, stride))

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for i in range(len(df)):

            frames = load_all_rgb_frames_from_how2sign(videos_folder, df.iloc[i]) # how2sign
            features = extract_features_fullvideo(i3d, frames, framespan, stride)

            video = str(df.iloc[i][0])
            print(i, video, len(features))

            name = video + '_' + str(df.iloc[i][1]).replace('.','-') + '_' + str(df.iloc[i][2]).replace('.','-')

            torch.save(features, os.path.join(outdir, name + '.pt'))

            with open('../all.txt', 'a') as f:
                info = "{\"ident\": \""+ name +"\", \"size\": " + str(len(features)) + "}"
                f.write(info + ",")
                f.close()


if __name__ == "__main__":
    weight = 'checkpoints/archive/nslt_2000_065538_0.514762.pt'

    # ======= Extract Features for PHEOENIX-2014-T ========
    videos_roots = [
        '../videos/train',
        '../videos/dev',
        '../videos/test'
    ]

    # out = '/home/dongxu/Dongxu/workspace/translation/data/PHOENIX-2014-T/features/i3d-features'
    out = '../i3d-features'

    run(weight, videos_roots, out, 'rgb')

    '''
    path_data = '../../input/h2s10/frases.txt'
    videos_folder = '../../input/h2s10/h2s10/'
    run_h2s(weight, path_data, videos_folder, out, 'rgb')
    '''