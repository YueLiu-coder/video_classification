"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os
import os.path
from subprocess import call
import cv2
import numpy as np


def extract_files():
    """After we have all of our videos split between train and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:

    [train|test], class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    print('ok')
    data_file = []
    folders = ['./train/', './test/']

    for folder in folders:
        class_folders = glob.glob(folder + '/*')

        for vid_class in class_folders:
            class_files = glob.glob(vid_class + '/*.avi')

            for video_path in class_files:
                # Get the parts of the file.
                video_parts = get_video_parts(video_path)

                train_or_test, classname, filename_no_ext, filename = video_parts

                # Only extract if we haven't done it yet. Otherwise, just get
                # the info.
                if not check_already_extracted(video_parts):
                    # Now extract it.
                    src = train_or_test + '/' + classname + '/' + \
                        filename
                    # dest = train_or_test + '/' + classname + '/' + \
                    #     filename_no_ext + '-%04d.jpg'
                    print(src)
                    # print(dest)

                    # call(["ffmpeg", "-i", src, dest])

                    # for i in range(1,10000):
                    #     file = dest[:-8]+'%04d.jpg'%i

                        # try:
                        #     img = cv2.imread(file)
                        #     imgd = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
                        #     imgu = cv2.resize(imgd, (img.shape[1], img.shape[0]))
                        #     err = cv2.absdiff(img, imgu)  # 差值的绝对值
                        #     cv2.imwrite(file, err)
                        #     # print(dest)
                        # except:
                        #     break



                # Now get how many frames it is.
                nb_frames = get_nb_frames_for_video(video_parts)
                # print([train_or_test, classname, filename_no_ext, nb_frames])
                nb_frames = extract_optical(vedio_path='./'+src,train_or_test=train_or_test,classes = classname,filename_no_ext=filename_no_ext)
                # print(nb_frames)
                data_file.append([train_or_test, classname, filename_no_ext, nb_frames])
                print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(train_or_test + '/' + classname + '/' +
                                filename_no_ext + '*.jpg')
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split("/")
    filename = parts[3]
    filename_no_ext = filename.split('.')[0]
    classname = parts[2]
    train_or_test = parts[1]

    return train_or_test, classname, filename_no_ext, filename

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    return False
    # return bool(os.path.exists(train_or_test + '/' + classname +
    #                            '/' + filename_no_ext + '-0001.jpg'))


def extract_optical(vedio_path='',train_or_test='train',classes = '',filename_no_ext=''):

    cap = cv2.VideoCapture(vedio_path)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    i = 1
    while (1):
        i = i + 1
        ret, frame2 = cap.read()
        if not ret:
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        filename = './{}_optimal/{}/{}-{:0>4}.jpg'.format(train_or_test,classes,filename_no_ext,i)
        dict = './{}_optimal/{}'.format(train_or_test,classes)
        if not os.path.exists(dict):
            os.mkdir(dict)  # 调用系统命令行来创建文件
        cv2.imwrite(filename, rgb)
        prvs = next
    cap.release()
    return i

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    extract_files()

if __name__ == '__main__':
    main()





