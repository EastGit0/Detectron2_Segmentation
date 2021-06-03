import numpy as np
import demo
import sys
import argparse
import os
import torch
import multiprocessing
import json
from predictor import VisualizationDemo
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

sys.path.insert(1,'/home/cs348k/pytorch_segmentation/')
# sys.path.insert(1,'/home/cs348k/pytorch_segmentation/classroom_trainer.py')
import dataloaders
import models
from classroom_trainer import ClassroomTrainer
from utils import losses
from utils import Logger
import time

# Checks folder where student places tensor representation of mask and corresponding image of frame and runs demo script on images to generate ground truths

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class Classroom_Process(multiprocessing.Process):
    def __init__(self, id, config, resume, queue):
        super(Classroom_Process, self).__init__()
        self.id = id
        self.config = config
        self.resume = resume
        self.queue = queue
        self.checkpoint_dir = os.path.join(self.config['trainer']['save_dir'], self.config['name'])
                 
    def run(self):
        print("Setting up classroom")

        # MODEL
        model = get_instance(models, 'arch', self.config, 81)
        print(f'\n{model}\n')

        # LOSS
        loss = getattr(losses, self.config['loss'])(ignore_index = self.config['ignore_index'])

        trainer = ClassroomTrainer(
                          model=model,
                          loss=loss,
                          resume=self.resume,
                          config=self.config,
                          train_loader=None,
                          val_loader=None,
                          train_logger=None)

        self.queue.put(True)

        while True:
            print("Waiting for Train signal")
            max_frame = self.queue.get()
            if max_frame > 0:
                print("Begin Training on JITNetX")

                self.config["train_loader"]["args"]["max_frame"] = max_frame

                # DATA LOADERS
                train_loader = get_instance(dataloaders, 'train_loader', self.config)

                weights_count = trainer.train(train_loader)
                self.queue.put(weights_count)
                del train_loader
                

def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = demo.get_parser().parse_args()
    cfg = demo.setup_cfg(args)

    predictor = VisualizationDemo(cfg)
    directory = args.input[0]
    count = 1
    next_path = args.input[0] + 'frame_' + str(count) + '.jpg'

    config = json.load(open(args.config_student))
    queue = multiprocessing.Queue()
    classroom = Classroom_Process(0, config, args.resume_student, queue)
    classroom.start()
    threshold = 0 # what should this be?
    
    ## Wait for Classroom
    # print("Waiting for classroom before starting")
    queue.get()
    print("Start Teacher Loop")

    #change this to a constant while loop
    while (True):
        if os.path.isfile(next_path):
            load_time = os.path.getsize(next_path)
            time.sleep(.25)
            while (load_time != os.path.getsize(next_path)):
                load_time = os.path.getsize(next_path)
                time.sleep(0.25)

            f = next_path
            img = read_image(f, format="BGR") 
            print("Image being segmented: " + f)
            # returns boolean value for if ground truth was generated and binary array for the mask being generated
            ground_truth_found, ground_truth_mask = predictor.run_on_image(img, count)
                
            # compare the ground truths generated above to the masks in the masks in /home/cs348k/data/student/masks
            if (not ground_truth_found):
                # remove original frame
                print("no ground truth generated for " + next_path)
                os.system('rm ' + next_path)
            else:
                if 0:
                  output_dir = args.output
                  #prediction = read_image(output_dir + 'masks/prediction_' + str(count))
                  prediction_tensor = torch.load(output_dir + 'predictions/prediction_' + str(count) + '.pt')
                  prediction_mask = prediction_tensor.cpu().numpy()
                  
                  # convert to np arrays
                  diff = ground_truth.astype(int) - prediction.astype(int)
                  raw_score = sum(sum(diff, []))
                
                # define threshold
                if count % 1024 == 0:
                # if (raw_score > threshold):
                    print("Retrain!!")

                    ## Signal Trainer to start work
                    queue.put(count)

                    ## Wait here now until training done?
                    weights_count = queue.get()

                    ## Delete Frames and masks used for training?
                    os.system("rm /home/cs348k/data/student/weights/{}/weights_{}.pth".format(config['arch']['type'], str(weights_count-1)))

                    ## Send weights to Local JITNet

            count += 1
            next_path = args.input[0] + 'frame_' + str(count) + '.jpg'
        
            
        # send updated weights to student network

if __name__ == '__main__':
    main()
