'''
code & scrap ML-related functions or objects
'''


from distutils.log import info
from re import sub
import smtplib
from email.mime.text import MIMEText
import os
import torch
import tqdm
# from google.colab import files # 설치안됨(m1?)


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


def colab_file_upload_drive():
    from google.colab import drive
    drive.mount("/content/gdrive")


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


def send_gmail(message: str = "default", subject: str = "default",
               from_gmail: str = "default",
               to_gmail: str = "default", login_gmail: str = "default",
                   app_password="default") -> None:
    info = {}
    info["message"] = "내용 : 본문내용 테스트입니다."
    info["from_gmail"] = "kanari2214@gmail.com"
    info["to_gmail"] = "whtnek@gmail.com"
    info["login_gmail"] = "kanari2214@gmail.com"
    info["subject"] = "제목 : 메일 보내기 테스트입니다."

    for i in info.keys():
        if locals()[i] != "default":
            info[i] = locals()[i]
    ######
    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.starttls()
    session.login('kanari2214@gmail.com', 'ckghunxkdnkujfdi')
    # message 설정
    msg = MIMEText(info["message"])
    msg['Subject'] = info["subject"]
    session.sendmail(info["from_gmail"], info["to_gmail"], msg.as_string())

    # 세션 종료
    session.quit()


def sample(x=""):
    if x is None:
        x = 'msg'
    return x


def pkl_dump(obj, save_path='./saved_data.pkl', verbose=True):
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)
    if verbose:
        var_str = var2str(obj)
        print(f'{var_str} - save complete!')


def pkl_load(load_path='./saved_data.pkl', verbose=True):
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    if verbose:
        name = load_path.split('/')[-1]
        print(f'{name} - load complete!')
    return data


def var2str(var_name):
    '''transform var_name to string'''
    return f'{var_name=}'.split('=')[0]


if __name__ == '__main__':
    # send_gmail(message='아하하 후하하하하하핳', subject="테스트메일임")
    import pickle
    x = {}
    x['mail'] = 'nate@gmail.com'
    x['name'] = 'nate'
    x['age'] = 29
    pkl_dump(x)
    print()
