import os

import numpy as np
import torch
import cv2

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


def load_all_rgb_frames_from_video(video, desired_channel_order='rgb'):
    cap = cv2.VideoCapture(video)
    
    frames = []
    faces = []
    
    currentFrame = 0
    while(True):
        try:
            ret, frame = cap.read()
            
            frame = cv2.resize(frame, dsize=(224, 224))




            ###
            cropped = crop_face(frame.copy())
            cropped = cv2.resize(cropped, dsize=(224, 224))

            if desired_channel_order == 'bgr':
                cropped = cropped[:, :, [2, 1, 0]]

            cropped = (cropped / 255.) * 2 - 1
            faces.append(cropped)
            ###



            
            #if desired_channel_order == 'bgr':
            #    frame = frame[:, :, [2, 1, 0]]

            #frame = (frame / 255.) * 2 - 1
            #frames.append(frame)

            currentFrame += 1
        except:
            break

    #nframes = np.asarray(frames, dtype=np.float32)
    nframes = []

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

    #print(ft)

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
    #i3d.replace_logits(1232)

    #print('loading weights {}'.format(weight))
    #i3d.load_state_dict(torch.load(weight))
    #i3d.load_state_dict(torch.load(weight)['ckpt'])

    i3d.cuda()
    #i3d = nn.DataParallel(i3d)

    print('feature extraction starts.')
    i3d.train(False)  # Set model to evaluate mode
    



    # Face model feature extractor
    #fmodel = InceptionI3d(400, in_channels=3)
    #fmodel.replace_logits(2000)
    #fmodel.cuda()
    #fmodel.train(False)




    # ===== extract features ======
    # for framespan, stride in [(4, 2), (16, 8), (32, 16)]:
    for framespan, stride in [(16, 2), (12, 2), (8, 2)]:
    # for framespan, stride in [(16, 8), (32, 16), (64, 32)]:

        outdir = os.path.join(outroot, 'span={}_stride={}'.format(framespan, stride))

        if not os.path.exists(outdir):
            os.makedirs(outdir)


        for ind, video in enumerate(videos):
            out_path = os.path.join(outdir, os.path.basename(video[:-4])) + '.pt'

            with open('./done.txt') as file:
                if out_path in file.read():
                    print('{} exists, continue'.format(out_path))
                    continue

            #if os.path.exists(out_path):
            #    print('{} exists, continue'.format(out_path))
            #    continue

            #if os.path.exists(out_path):
            #    print('{} exists, continue'.format(out_path))
            #    done.append(out_path)
            #    continue
            #with open('./done.txt', 'a') as f:
            #    f.writelines('\n'.join(done))
            #while(True):
            #    print('opa')

            frames, face_frames = load_all_rgb_frames_from_video(video, inp_channels)
            #features = extract_features_fullvideo(i3d, frames, framespan, stride)

            face_features = extract_features_fullvideo(i3d, face_frames, framespan, stride)

            #for i in range(len(face_features)):
            #    features.append(face_features[i])

            if ind % 1 == 0:
                #print(ind, video, len(features), features[0].shape)
                print(ind, video, len(face_features), face_features[0].shape)

            torch.save(face_features, os.path.join(outdir, os.path.basename(video[:-4])) + '.pt')

            with open('./done.txt', 'a') as f:
                f.write(out_path + "\n")
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
