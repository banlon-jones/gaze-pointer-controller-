import os
import sys
import time
import cv2


from collections import namedtuple
from argparse import ArgumentParser
import logging as log
from demoTools import demoutils
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FaceLandmarksDetection
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController

MyStruct = namedtuple("shoppingInfo", "shopper, looker")
INFO = MyStruct(0, 0)
POSE_CHECKED = False
DELAY = 5

def args_parser():
    """
    Parse command line arguments.
    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fm", "--facemodel", required=True,
                        help="Path to an .xml file with a pre-trained"
                        "face detection model")
    parser.add_argument("-pm", "--posemodel", required=True,
                        help="Path to an .xml file with a pre-trained model"
                        "head pose model")
    parser.add_argument("-lm", "--landmarksmodel", required=True,
                        help="Path to an .xml file with a pre-trained model"
                        "landmarks model")
    parser.add_argument("-gm", "--gazemodel", required=True,
                        help="Path to an .xml file with a pre-trained model"
                        "gaze estimation model")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or image."
                        "'cam' for capturing video stream from camera")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                        "path to a shared library with the kernels impl.")
    parser.add_argument("-d", "--device", default="CPU", type=str,
                        help="Specify the target device to infer on; "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. Looks"
                        "for a suitable plugin for device specified"
                        "(CPU by default)")
    parser.add_argument("-c", "--confidence", default=0.5, type=float,
                        help="Probability threshold for detections filtering")
    parser.add_argument("-o", "--output_dir", help = "Path to output directory", type = str, default = None)
    parser.add_argument("-v", "--visualization", default=None, type=str,
                        help="Select between fm,lm,pm,gm,")

    return parser


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    global INFO
    global DELAY
    global POSE_CHECKED
    controller = MouseController()

    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = args_parser().parse_args()
    logger = log.getLogger()

    if args.input == 'cam':
       input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    initial_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(os.path.join(args.output_dir, "shopper.mp4"), cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_width, initial_height), True)
    frame_count = 0

    job_id = 1
    progress_file_path = os.path.join(args.output_dir,'i_progress_'+str(job_id)+'.txt')

    infer_time_start = time.time()

    if input_stream:
        cap.open(args.input)
        # Adjust DELAY to match the number of FPS of the video file
        DELAY = 1000 / cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        logger.error("ERROR! Unable to open video source")
        return

    # Initialise the class
    if args.cpu_extension:
        facedet = FaceDetection(args.facemodel, args.confidence,extensions=args.cpu_extension)
        posest = HeadPoseEstimation(args.posemodel, args.confidence,extensions=args.cpu_extension)
        landest = FaceLandmarksDetection(args.landmarksmodel, args.confidence,extensions=args.cpu_extension)
        gazeest = GazeEstimation(args.gazemodel, args.confidence,extensions=args.cpu_extension)
    else:
        facedet = FaceDetection(args.facemodel, args.confidence)
        posest = HeadPoseEstimation(args.posemodel, args.confidence)
        landest = FaceLandmarksDetection(args.landmarksmodel, args.confidence)
        gazeest = GazeEstimation(args.gazemodel, args.confidence)

    # Load the network to IE plugin to get shape of input layer

    facedet.load_model()
    posest.load_model()
    landest.load_model()
    gazeest.load_model()
    print("loaded models")
    ret, frame = cap.read()
    while ret:
        looking = 0
        POSE_CHECKED = False
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            print ("checkpoint *BREAKING")
            break

        if frame is None:
            log.error("checkpoint ERROR! blank FRAME grabbed")
            break

        initial_width = int(cap.get(3))
        initial_height = int(cap.get(4))

        # Start asynchronous inference for specified request
        inf_start_fd = time.time()
        # Results of the output layer of the network
        coords, frame = facedet.predict(frame)
        if args.visualization == "fm":
            cv2.startWindowThread()
            cv2.namedWindow("preview")
            cv2.imshow("preview", frame)
        det_time_fd = time.time() - inf_start_fd
        if len(coords) > 0:
            [xmin, ymin, xmax, ymax] = coords[0]  # use only the first detected face
            head_pose = frame[ymin:ymax, xmin:xmax]
            inf_start_hp = time.time()
            is_looking, pose_angles = posest.predict(head_pose)
            if args.visualization == "pm":
                cv2.startWindowThread()
                cv2.namedWindow("preview")
                p = "Pose Angles {}, is Looking? {}".format(pose_angles,is_looking)
                cv2.putText(frame, p, (50, 15), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (255, 255, 255), 1)
                cv2.imshow("preview", frame)

            if is_looking:
                det_time_hp = time.time() - inf_start_hp
                POSE_CHECKED = True
                print(is_looking)
                inf_start_lm = time.time()
                coords,f = landest.predict(head_pose)
                if args.visualization == "lm":
                    cv2.startWindowThread()
                    cv2.namedWindow("preview")
                    cv2.imshow("preview", f)

                frame[ymin:ymax, xmin:xmax] = f
                det_time_lm = time.time() - inf_start_lm
                [[xlmin, ylmin, xlmax, ylmax], [xrmin, yrmin, xrmax, yrmax]] = coords
                left_eye_image = f[ylmin:ylmax, xlmin:xlmax]
                right_eye_image = f[yrmin:yrmax, xrmin:xrmax]
                output,gaze_vector = gazeest.predict(left_eye_image,right_eye_image,pose_angles)
                if args.visualization == "gm":
                    cv2.startWindowThread()
                    cv2.namedWindow("preview")
                    p = "Gaze Vector {}".format(gaze_vector)
                    cv2.putText(frame, p, (50, 15), cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (255, 255, 255), 1)
                    fl = draw_gaze(left_eye_image, gaze_vector)
                    fr = draw_gaze(right_eye_image, gaze_vector)
                    f[ylmin:ylmax, xlmin:xlmax] = fl
                    f[yrmin:yrmax, xrmin:xrmax] = fr
                    cv2.imshow("preview", f)

                if frame_count%10 == 0:
                    controller.move(output[0],output[1])
        # Draw performance stats
        inf_time_message = "Face Inference time: {:.3f} ms.".format(det_time_fd * 1000)
    #
        if POSE_CHECKED:
            cv2.putText(frame, "Head pose Inference time: {:.3f} ms.".format(det_time_hp * 1000), (0, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, inf_time_message, (0, 15), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (255, 255, 255), 1)
        out.write(frame)
        if frame_count%10 == 0:
            demoutils.progressUpdate(progress_file_path, int(time.time()-infer_time_start), frame_count, video_len)
        if args.output_dir:
            total_time = time.time() - infer_time_start
            with open(os.path.join(args.output_dir, 'stats.txt'), 'w') as f:
                f.write(str(round(total_time, 1))+'\n')
                f.write(str(frame_count)+'\n')
    facedet.clean()
    posest.clean()
    landest.clean()
    gazeest.clean()
    cap.release()
    cv2.destroyAllWindows()
    out.release()


def draw_gaze(screen_img, gaze_pts, gaze_colors=None, scale=10, return_img=False, cross_size=16, thickness=10):

    """ Draws an "x"-shaped cross on a screen for given gaze points, ignoring missing ones
    """
    width = int(cross_size * scale)

    draw_cross(screen_img, gaze_pts[0] * scale, gaze_pts[1] * scale, (0, 255, 255), width, thickness)
    return  screen_img

def draw_cross(bgr_img,x, y,color=(255, 255, 255), width=2, thickness=1):

    """ Draws an "x"-shaped cross at (x,y)
    """

    x, y, w = int(x), int(y), int(width / 2)

    cv2.line(bgr_img, (x - w , y - w), (x + w , y + w), color, thickness)
    cv2.line(bgr_img, (x - w , y + w), (x + w, y - w), color, thickness)

if __name__ == '__main__':
    try:
        main()
    except:
        sys.exit()

