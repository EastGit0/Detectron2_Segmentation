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

        # DATA LOADERS
        train_loader = get_instance(dataloaders, 'train_loader', self.config)
        # val_loader = get_instance(dataloaders, 'val_loader', self.config)

        # MODEL
        model = get_instance(models, 'arch', self.config, train_loader.dataset.num_classes)
        print(f'\n{model}\n')

        # LOSS
        loss = getattr(losses, self.config['loss'])(ignore_index = self.config['ignore_index'])

        trainer = ClassroomTrainer(
                          model=model,
                          loss=loss,
                          resume=self.resume,
                          config=self.config,
                          train_loader=train_loader,
                          val_loader=None,
                          train_logger=None)

        while True:
            print("Waiting for Train signal")
            start_training = self.queue.get()
            if start_training:
                print("Begin Training on JITNetX")
                weights_count = trainer.train()
                self.queue.put(weights_count)
                

                


def main():
    args = demo.get_parser().parse_args()
    cfg = demo.setup_cfg(args)

    predictor = VisualizationDemo(cfg)
    directory = args.input[0]
    count = 0
    next_path = args.input[0] + 'frame_' + str(count + 1) + '.jpg'

    config = json.load(open(args.config_student))
    queue = multiprocessing.Queue()
    classroom = Classroom_Process(0, config, args.resume_student, queue)
    threshold = 0 # what should this be?
    
    #change this to a constant while loop
    while (True):
        if 0:
        # if os.path.isfile(next_path):
            f = next_path
            img = read_image(f, format="BGR") 
            print("Image being segmented: " + f)
            # returns boolean value for if ground truth was generated and binary array for the mask being generated
            ground_truth_found, ground_truth_mask = predictor.run_on_image(img, count)
                
            # compare the ground truths generated above to the masks in the masks in /home/cs348k/data/student/masks
            if (not ground_truth_found):
                # remove original frame
                print("no ground truth generated for " + next_path)
                #os.system('rm ' + next_path)
            else:
                output_dir = args.output
                #prediction = read_image(output_dir + 'masks/prediction_' + str(count))
                prediction_tensor = torch.load(output_dir + 'masks/prediction_' + str(count) + '.pt')
                prediction_mask = prediction_tensor.cpu().numpy()
                
                # convert to np arrays
                diff = ground_truth.astype(int) - prediction.astype(int)
                raw_score = sum(sum(diff, []))
                
                # define threshold
                if (raw_score > threshold):
                    print("Retrain!!")

                    ## Signal Trainer to start work
                    queue.put(True)

                    ## Wait here now until training done?
                    weights_count = queue.get()

                    ## Delete Frames and masks used for training?

                    ## Send weights to Local JITNet
                    # self.ssh_weights = SSHClient()
                    # self.ssh_weights.load_system_host_keys()
                    # self.ssh_weights.connect('35.233.229.168')
                    # self.scp_weights = SCPClient(self.ssh_weights.get_transport())
                    # self.scp_weights.put("weights_{}.pth".format(weights_count), remote_path='/home/cs348k/data/student/frames')

            count += 1
            next_path = args.input[0] + 'frame_' + str(count) + '.jpg'
        
            
        # send updated weights to student network

if __name__ == '__main__':
    main()
