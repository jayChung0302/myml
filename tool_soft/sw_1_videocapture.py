# read video file and capture frames
import cv2


def get_cam():
    capture = cv2.VideoCapture(0)
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)

    print(f"CV_CAP_PROP_FRAME_WIDTH: {frame_width}")
    print(f"CV_CAP_PROP_FRAME_HEIGHT: {frame_height}")
    print(f"CAP_PROP_FPS: {fps}")

    if capture.isOpened() is False:
        print("Error opening the camera")

    while capture.isOpened():
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        if ret:
            cv2.imshow('Input frame from the camera', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):  # press q to quit
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    get_cam()
