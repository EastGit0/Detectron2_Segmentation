import numpy as np
import demo
import sys
import argparse
import os
from predictor import VisualizationDemo
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

# Checks folder where student places tensor representation of mask and corresponding image of frame and runs demo script on images to generate ground truths


def main():
    args = demo.get_parser().parse_args()
    cfg = demo.setup_cfg(args)

    predictor = VisualizationDemo(cfg)
    directory = args.input[0]
    count = 0
    next_path = args.input[0] + 'frame_' + str(count + 1) + '.jpg'

    #change this to a constant while loop
    while (True):
        if os.path.exists(next_path):
            f = next_path
            # checking if it is a file
            if os.path.isfile(f):
                img = read_image(f, format="BGR") 
                print("Image being segmented: " + f)
                predictions, visualized_output = predictor.run_on_image(img, count)
                # compare the ground truths generated above to the masks in the masks in /home/cs348k/data/student/masks

            output_dir = args.output
            ground_truth_path = output_dir + '/ground_truths/ground_truth_' + str(count) + '.png'
            print("Ground truth path: " + ground_truth_path)
            ground_truth = read_image(ground_truth_path)

            # student_mask = read_image(output_dir + 'masks/student_mask_' + str(count))
            # update count after files are retrieved
            count += 1
            next_path = args.input[0] + 'frame_' + str(count) + '.jpg'
        
            # convert to np arrays
            #diff = ground_truth - student_mask
            #raw_score = sum(sum(diff, []))
        
            # define threshold
            #if (raw_score > threshold):
            # re train student network on Detectron ground truth masks

        # send updated weights to student network

if __name__ == '__main__':
    main()
