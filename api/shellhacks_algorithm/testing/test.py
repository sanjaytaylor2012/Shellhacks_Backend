import matplotlib.pyplot as plt
import cv2
import numpy as np
from data_curation.pose_estimation import VideoProcessor
from algorithm.matching_angle_nets import evaluate_sequence
import os


def main():
    VideoProcessor.process_video('../data/basketball/basketball_side_view.mp4')

    # videos = list(os.walk("../data"))[1:]
    #
    # for dir, _, files in videos:
    #     for file in files:
    #         print(dir + file)
    #         VideoProcessor.process_video(dir + '/' + file, compression=4)


if __name__ == "__main__":
    main()
