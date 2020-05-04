import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin
import os
import cv2
import sys

from pathlib import Path
import math
sys.path.insert(0, str(Path().resolve().parent.parent))

'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
Since you will be using four models to build this project, you will need to replicate this file
for each of the models.

This has been provided just to give you an idea of how to structure your model class.
'''

class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, threshold, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_ob = None
        self.input_ob1 = None
        self.input_ob2 = None
        self.output_ob = None
        self.threshold = threshold
        self.device = device
        self.model_name = model_name
        self.extensions = extensions
        self.initial_width = None
        self.initial_height = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        model_xml = self.model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Initialize the plugin

        self.plugin = IEPlugin(device=self.device)
        # Add a CPU extension, if applicable
        if self.extensions and "CPU" in self.device:
            self.plugin.add_cpu_extension(self.extensions)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        # Check for any unsupported layers, and let the user know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.network.layers.keys() if l not in self.plugin.get_supported_layers(self.network)]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load(self.network)

        # Get the input layer
        self.input_ob = next(iter(self.network.inputs))
        self.output_ob = next(iter(self.network.outputs))

    def check_plugin(self, plugin):
        '''
        TODO: You will need to complete this method as a part of the
        standout suggestions

        This method checks whether the model(along with the plugin) is supported
        on the CPU device or not. If not, then this raises and Exception
        '''
        raise NotImplementedError

    def predict(self,left_eye_image,right_eye_image, pose_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        count = 0
        coords = None
        self.initial_width = left_eye_image.shape[1]
        self.initial_height = left_eye_image.shape[0]
        left_eye_image, right_eye_image = self.preprocess_input(left_eye_image, right_eye_image)
        self.exec_network.requests[0].async_infer(inputs={"head_pose_angles": pose_angles, "left_eye_image": left_eye_image, "right_eye_image":right_eye_image})
        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_ob]
            out = self.preprocess_output(left_eye_image, right_eye_image, pose_angles, outputs)
            return out

    def preprocess_input(self, left_eye_image,right_eye_image):
        '''
        TODO: You will need to complete this method.
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # print(self.network.inputs[self.input_blob].shape)
        # (n, c, h, w) = self.network.inputs[self.input_blob].shape

        left_eye_image = cv2.resize(left_eye_image, (60, 60))
        left_eye_image = left_eye_image.transpose((2,0,1))
        left_eye_image = left_eye_image.reshape((1, 3, 60, 60))

        right_eye_image = cv2.resize(right_eye_image, (60, 60))
        right_eye_image = right_eye_image.transpose((2,0,1))
        right_eye_image = right_eye_image.reshape((1, 3, 60, 60))

        return left_eye_image,right_eye_image

    def preprocess_output(self, left_eye_image, right_eye_image, pose_angles, outputs):
        '''
        TODO: You will need to complete this method.
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze_vector = outputs[0]
        roll = gaze_vector[2]#pose_angles[0][2][0]
        cs = math.cos(roll * math.pi / 180.0)
        sn = math.sin(roll * math.pi / 180.0)

        tmpX = gaze_vector[0] * cs + gaze_vector[1] * sn
        tmpY = -gaze_vector[0] * sn + gaze_vector[1] * cs

        return (tmpX,tmpY),(gaze_vector)
        # raise NotImplementedError

    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.plugin
        del self.network
        del self.exec_network
