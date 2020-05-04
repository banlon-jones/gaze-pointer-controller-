*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Computer Pointer Controller

Computer Pointer Controller is a human computer interaction application using computer vision algorithms for controlling the mouse pointer by calculating a person's point of gaze. It explores potential of eye gaze as a pointing device.

This project using a pipeline of a four models we create an application to control the computer mouse with the our face's gaze.
There are four models: Face Detection, head Pose Estimation, Facial Landmarks detection for locating the eyes in the face and lastly, the gaze estimation which gives us the gaze vector for controlling the mouse


## Project Set Up and Installation
first download and install intel OpenVino toolkit `https://docs.openvinotoolkit.org/latest/index.html`
#### Enable the virtual environment:

```
source /opt/intel/openvino/bin/setupvars.sh
```

#### Install the requirements:

```
pip install -r requirements.txt
```

We then have to download all the models. We do this on the project's folder. So, cd in the folder and run the following commands:

``` 
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 
 ```

## Demo
For a demo of the app `cd` into the `src` folder first.

Run the following command:

#### for FP32

```

python3 main.py -fm ../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -pm ../models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -lm ../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -gm ../models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i '../bin/demo.mp4' -o . -d "CPU" -c 0.5
```

#### for FP16

```
python3 main.py -fd ../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -hpe ../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -fld ../intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -ge ../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i '../bin/demo.mp4' -o . -d "CPU" -c 0.5
```

#### for INT8 model

```
python3 main.py -fd ../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -hpe ../models/intel/head-pose-estimation-adas-0001/FP32-INT8/head-pose-estimation-adas-0001.xml -fld ../models/intel/landmarks-regression-retail-0009/FP32-INT8/landmarks-regression-retail-0009.xml -ge ../models/intel/gaze-estimation-adas-0002/FP32-INT8/gaze-estimation-adas-0002.xml -i '../bin/demo.mp4' -o . -d "CPU" -c 0.5
```

## Documentation
There are four main command line arguments that must be passed for the project to run successfully and they are the arguments that specify the paths to the model `.xml` files.
​
- `-fd` specifies the path to the face detection model
- `-hpe` specifies the path to the head pose estimation model.
- `-fld` specifies the path to the facial landmarks detection model.
- `-ge` specifies the path to the gaze estimation model.
- `-i`  specifies the path to the input video stream. If you wish to use video feat from your camera or webcam, use `cam('-i')` instead.
- To visualize any of the models while the project is running, include the model (either fd, hpe, fld or ge) after the `-v` argument.

## Benchmarks
As seen when the project is run, the benchmarks are as follows:
​
- For FP32 models, I had:
​
```
Inference time for Face Detection Model: 
```
​
```
Inference time for Head Pose Estimation Model: 
```
​
```
Inference time for Facial Landmarks detection Model: 
```
​
```
Total input/output processing time: 
```
​
```
Frame count: 
```
​
​
- For FP16 models, I had:
​
```
Inference time for Face Detection Model: 
```
​
```
Inference time for Head Pose Estimation Model: 
```
​
```
Inference time for Facial Landmarks detection Model: 
```
​
```
Total input/output processing time: 
```
​
```
Frame count: 
```
​
- For FP32-INT8 models, I had:
​
```
Inference time for Face Detection Model: 
```
​
```
Inference time for Head Pose Estimation Model: 
```
​
```
Inference time for Facial Landmarks detection Model: 
```
​
```
Total input/output processing time: 
```
​
```
Frame count: 
```
​
​
##### Note: These benchmarks may be slightly different on your own system.
## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

### Async Inference
*TODO (Optional):* If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
*TODO (Optional):* There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
