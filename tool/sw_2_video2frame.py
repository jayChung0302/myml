from webbrowser import get
import cv2
import os


def get_frames(filepath, save_dir='./output'):
    video = cv2.VideoCapture(filepath)

    if not video.isOpened():
        print("Could not Open :", filepath)
        exit(0)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    print("length :", length)
    print("width :", width)
    print("height :", height)
    print("fps :", fps)

    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError:
        print(f'Failed to create directory in {save_dir}')
    count = 0

    while(True):
        ret, image = video.read()
        if ret:
            cv2.imwrite(f'{save_dir}/images/frame_{str(count).zfill(6)}.jpg', image)
            print('Saved frame number :', str(int(video.get(1))))
            count += 1
        if count == length:
            break
    video.release()


if __name__ == '__main__':
    get_frames('/Users/chung/Downloads/myroom.mov')
