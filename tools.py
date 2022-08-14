'''
code & scrap ML-related functions or objects
'''


from distutils.log import info
from re import S, sub
from select import kevent
import smtplib
from email.mime.text import MIMEText
import os
from glob import glob
from typing import List
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
import cv2
import numpy as np
from omegaconf import OmegaConf
import yaml
from datetime import datetime
from PIL import Image
import tqdm
# from google.colab import files # 설치안됨(m1?)


def get_today() -> List:
    '''Get today's year, month, day information list'''
    return list(map(int, datetime.today().strftime("%Y %m %d").split()))


def get_weeknum():
    '''Get today's # of week'''
    date = datetime(*get_today())
    return date.isocalendar()[1]


def pad_np_img_to_modulo(img, mod):
    '''(np)C x H x W image padding'''
    channels, height, width = img.shape
    out_height = ceil_to_modulo(height, mod)
    out_width = ceil_to_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


def ceil_to_modulo(x, mod):
    '''Return ceiled x to mod-able form with mod param'''
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def get_config_from_yaml(config_path):
    '''Get config information from yaml'''
    with open(config_path, 'r') as f:
        config = OmegaConf.create(yaml.safe_load(f))
    return config


def get_img_dir(path_dir='./', exts=['*.jpg', '*.png']):
    '''Get image data dir from given path'''
    data_dirs = []
    for ext in exts:
        data_dirs += glob(os.path.join(path_dir, ext))
    return data_dirs


def calculate_mean_var(dataloader):
    '''Calculate mean and var from dataloader'''

    total_sum = torch.zeros(3)
    total_sum_sq = torch.zeros(3)
    idx = 0
    for inputs, *label in dataloader:
        if idx == 0:
            print('the input size is: ', inputs.size())
            *out, img_height, img_width = inputs.size()
        if len(out) > 1:
            total_sum += inputs.sum(axis=(0, 2, 3))
            total_sum_sq += (inputs ** 2).sum(axis=(0, 2, 3))
        else:
            total_sum += inputs.sum(axis=(1, 2))
            total_sum_sq += (inputs ** 2).sum(axis=(1, 2))

        idx += 1
    total_num = len(dataloader) * img_height * img_width
    total_mean = total_sum / total_num
    total_var = total_sum_sq / total_num - total_mean ** 2
    total_std = torch.sqrt(total_var)
    return total_mean, total_std


def make_beep_sound_intel_mac(phrase: str):
    # make sound in mac!!
    os.system(f'say {phrase}')

    import os
    os.system('afplay /System/Library/Sounds/Sosumi.aiff')
    return


def calculate_kl_loss(y_true, y_pred):
    '''Calculate KL-divergence loss manually'''
    loss = y_true * torch.log(y_true / y_pred)
    return loss


def mkdir(dir_path, exist_ok=False):
    '''Make dir except for not exist'''
    os.makedirs(dir_path, exist_ok=exist_ok)


def torch_to_image(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def numpy_to_image():
    pass


def load_np_img(fname: str, mode='RGB', return_orig=False):
    '''Load image(numpy) from filename(string)'''
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img


def normalize():
    pass


def split_file_name(filename):
    '''Get splited file name path, name, extension'''
    if len(filename.split('.')) > 2:
        path = os.path.dirname(filename)
        name = os.path.basename(filename)[:-8]
        extension = os.path.basename(filename)[-7:]
    else:
        path = os.path.dirname(filename)
        name, extension = os.path.basename(filename).split('.')
    return path, name, extension


def show_path(path=''):
    '''Show all image in the path'''
    files = glob.glob(f'{path}/*')

    files_ = sorted(files)

    for idx, file in enumerate(files_):
        path, name, ext = split_file_name(file)
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        cv2.imshow(name, img)
        cv2.moveWindow(name, 40, 30)
        keycode = cv2.waitKey()  # 키보드 입력 반환 값 저장

        if keycode == ord('q'):  # i 또는 I 누르면 영상 반전
            cv2.destroyAllWindows()
        else:
            cv2.destroyAllWindows()


def repeat(message: str, times: int = 2) -> list:
    return [message] * times


def send_gmail(subject: str = "default", message: str = "default",
               from_gmail: str = "default",
               to_gmail: str = "default", login_gmail: str = "default",
                   app_password="default") -> None:
    """Send gmail with given string"""
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
    ######
    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.starttls()

    session.login(info["login_gmail"], info["app_password"])
    # message 설정
    msg = MIMEText(info["message"])
    msg['Subject'] = info["subject"]
    session.sendmail(info["from_gmail"], info["to_gmail"], msg.as_string())

    # session end
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
    '''load data from dumped pickle'''
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
    print(get_weeknum())

    x = torch.randn(32)
    x = torch.clip(x, 0, 1)
    print(x)
    # mkdir('./ex_folder')

    with open('./keys_hash/kanari.txt') as f:
        a = f.readlines()
        print(a[-1].split('\n')[0])
    # send_gmail('급여명세서 줘.', '여깄습니다')

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
    }
    data_dir = 'data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    print(calculate_mean_var(image_datasets['train']))
    print(calculate_mean_var(image_datasets['val']))
