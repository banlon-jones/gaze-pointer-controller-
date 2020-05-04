import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin
import os
import cv2
# import argpar
import time
import sys
from argparse import ArgumentParser
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))

'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
Since you will be using four models to build this project, you will need to replicate this file
for each of the models.

This has been provided just to give you an idea of how to structure your model class.
'''

class FaceLandmarksDetection:

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


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        count = 0
        coords = None
        self.initial_width = image.shape[1]
        self.initial_height = image.shape[0]
        frame = self.preprocess_input(image)
        self.exec_network.requests[0].async_infer(inputs={self.input_ob: frame})
        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_ob]
            frame,coords = self.preprocess_output(image, outputs)
            return coords, frame

    def check_plugin(self, plugin):
        '''
        TODO: You will need to complete this method as a part of the
        standout suggestions

        This method checks whether the model(along with the plugin) is supported
        on the CPU device or not. If not, then this raises and Exception
        '''
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        TODO: You will need to complete this method.
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        (n, c, h, w) = self.network.inputs[self.input_ob].shape
        frame = cv2.resize(image, (w, h))
        frame = frame.transpose((2,0,1))
        frame = frame.reshape((n, c, h, w))
        return frame

    def preprocess_output(self, frame, outputs):
        '''
        TODO: You will need to complete this method.
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        current_count = 0
        coords = []
        # frame = cv2.UMat(frame).get()
        outputs= outputs[0]
        xl,yl = outputs[0][0] * self.initial_width, outputs[1][0] * self.initial_height
        xr,yr = outputs[2][0] * self.initial_width, outputs[3][0] * self.initial_height

        xlmin = xl-25
        ylmin = yl-25
        xlmax = xl+25
        ylmax = yl+25

        xrmin = xr-25
        yrmin = yr-25
        xrmax = xr+25
        yrmax = yr+25

        cv2.rectangle(frame, (xlmin, ylmin), (xlmax, ylmax), (0, 55, 255), 1)
        cv2.rectangle(frame, (xrmin, yrmin), (xrmax, yrmax), (0, 55, 255), 1)
        coords = [[int(xlmin),int(ylmin),int(xlmax),int(ylmax)],[int(xrmin),int(yrmin),int(xrmax),int(yrmax)]]
        return frame, coords

    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.plugin
        del self.network
        del self.exec_network
