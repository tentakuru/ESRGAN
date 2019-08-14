import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch
import os

#model_path = sys.argv[1]  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
models = []
#model_path = "C:\\mxnet\\models\\SRGAN\\"
model_path = "C:\\mxnet\\models\\SRGAN\\models\\"
#model_path = "C:\\mxnet\\models\\EDVR\\"
for model in os.listdir(model_path):
    if not model.startswith("1x"):
        #if model.startswith("4x_Faces_N_250000"):
        models.append(model_path + model)
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

#test_img_folder = 'LR/*'
#test_img_folder = sys.argv[1] + "\\*"


modelindex = 8
#3 is best, 12 pretty nice, 14 nice but strong, 0 nice

model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(models[modelindex]), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(models[modelindex]))

testpre = []
for folder in os.listdir("F:\\JavPlayer v1.03_win64_Nvidia\\TG\\"):
    if folder.startswith("input"):
        testpre.append(folder)



idx = 0
myi = 0
try:
    i = sys.argv[2]
except:
    i = 0
if i >= 1:
    pass
else:
    for thing in testpre:
        try:
            os.mkdir(os.getcwd() + "\\" + "output" + str(myi))
        except:
            pass
        test_img_folder = os.getcwd() + "\\input" + str(myi) + "\\*"
        for path in glob.glob(test_img_folder):
            if path.endswith(".png"):
                idx += 1
                base = os.path.splitext(os.path.basename(path))[0]
                print(idx, base)
                # read image
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = img * 1.0 / 255
                img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
                img_LR = img.unsqueeze(0)
                img_LR = img_LR.to(device)

                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                output = (output * 255.0).round()
                cv2.imwrite(os.getcwd() + "\\" + 'output' + str(myi) + '/output_{:s}.png'.format(base), output)
        myi+=1
