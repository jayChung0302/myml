'''
code & scrap ML-related functions or objects
'''


from distutils.log import info
from re import sub
from select import kevent
import smtplib
from email.mime.text import MIMEText
import os
from glob import glob
import torch
import cv2
import numpy as np
from omegaconf import OmegaConf
import yaml
from PIL import Image
import tqdm
# from google.colab import files # 설치안됨(m1?)

def load_np_img(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img

def get_data_dir(path_dir='./', exts=['*.jpg', '*.png']):
    data_dirs = []
    for ext in exts:
        data_dirs += glob(os.path.join(path_dir, ext))
    return data_dirs


def calculate_mean_var(dataloader):
    _, _, img_height, img_width = dataloader[0].size()

    total_sum = torch.zeros(3)
    total_sum_sq = torch.zeros(3)
    for inputs in tqdm(dataloader):
        total_sum += inputs.sum(axis=(0, 2, 3))
        total_sum_sq += (inputs ** 2).sum(axis=(0, 2, 3))
    total_num = len(dataloader) * img_height * img_width
    total_mean = total_sum / total_num
    total_var = total_sum_sq / total_num - total_mean ** 2
    total_std = torch.sqrt(total_var)
    return total_mean, total_std


def colab_file_upload():
    uploaded = files.upload()


def colab_file_upload_drive(mount_dir="/content/gdrive"):
    from google.colab import drive
    drive.mount(mount_dir)


def make_beep_sound_intel_mac(phrase: str):
    # make sound in mac!!
    os.system(f'say {phrase}')

    import os
    os.system('afplay /System/Library/Sounds/Sosumi.aiff')
    return


def calculate_kl_loss(y_true, y_pred):
    # calculate KL-divergence loss
    loss = y_true * torch.log(y_true / y_pred)
    return loss


def mkdir(dir_path, exist_ok=False):
    os.makedirs(dir_path, exist_ok=exist_ok)

def ransac():
    pass


def torch_to_image():
    pass


def numpy_to_image():
    pass


def normalize():
    pass


def plot_and_saving():
    pass


def glob_png_and_jpg():
    pass


def repeat(message: str, times: int = 2) -> list:
    return [message] * times


def send_gmail(subject: str = "default", message: str = "default",
               from_gmail: str = "default",
               to_gmail: str = "default", login_gmail: str = "default",
                   app_password="default") -> None:
    info = {}
    info["message"] = "내용 : 본문내용 테스트입니다."
    info["from_gmail"] = "kanari2214@gmail.com"
    info["to_gmail"] = "whtnek@gmail.com"
    info["subject"] = "제목 : 메일 보내기 테스트입니다."
    info["app_password"] = ""
    info["login_gmail"] = ""

    for i in info.keys():
        if locals()[i] != "default":
            info[i] = locals()[i]
        else:
            try:
                if (i == "app_password") or (i == "login_gmail"):
                    with open('./keys_hash/kanari.txt') as f:
                        x = f.readlines()
                        id = x[0].split('\n')[0]
                        pwd = x[-1].split('\n')[0]
                        info["login_gmail"] = id
                        info["app_password"] = pwd
            except:
                print('You have to specify app password.')
    print(info.keys())
    ######
    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.starttls()

    session.login(info["login_gmail"], info["app_password"])
    # message 설정
    msg = MIMEText(info["message"])
    msg['Subject'] = info["subject"]
    session.sendmail(info["from_gmail"], info["to_gmail"], msg.as_string())

    # 세션 종료
    session.quit()


def pkl_dump(obj, save_path='./', name=None, verbose=True):
    '''pkl_dump(var_name, name=f'{var_name=}'.split('=')[0])'''
    import pickle
    if name:
        save_path = save_path + name + '.pkl'
    else:
        save_path = save_path + 'saved_data.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)
    if verbose:
        print(f'save complete {save_path}!')


def pkl_load(load_path='./saved_data.pkl', verbose=True):
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    if verbose:
        name = load_path.split('/')[-1]

        print(f'{name} - load complete!')
    return data


if __name__ == '__main__':
    import pickle
    x = {}
    x['mail'] = 'nate@gmail.com'
    x['name'] = 'nate'
    x['age'] = 29
    print(f'{x=}'.split('=')[0])
    print(f'{x=}')
    pkl_dump(x, name=f'{x=}'.split('=')[0])
    # with open('./keys_hash/kanari.txt') as f:
    #     a = f.readlines()
    #     print(a[-1].split('\n')[0])
    mkdir('./ex_folder')
    # send_gmail('ex_subtitle.', 'ex_content')
    