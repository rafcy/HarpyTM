# Traffic Monitoring Application in Python - Harpy Dataset Source Code.

**Project Website link - https://www.kios.ucy.ac.cy/harpydata/**

![Demo Video of this Proejct](https://www.youtube.com/embed/7X2afxxGR4M)


**Repository initially published here https://codeocean.com/capsule/6749177/tree.**

The main purpose of the application is to extract traffic data from vehicles on roads using aerial footage taken from static UAVs. To process the footage, deep neural network detector is used (YOLO) alongside with the OpenCV library in ordered to be executed in python. Furthermore, multiple algorithms are used, such as Kalman, Hungarian, in order to match the detections between sequential frames and extract the vehicles and their trajectories. Hence, the velocities and the moving direction of the vehicles are also calculated for each vehicle for every frame. More information about the algorithms used can be found in the 'Extracting the fundamental diagram from aerial footage' paper cited below.

Our paper introducing the Harpy dataset and the used methods is published at the IEEE VTC 2020 and available here, https://ieeexplore.ieee.org/document/9128534.
A preprint on arXiv.org is available here,https://arxiv.org/abs/2007.03227. 

##### To reference the dataset or the source code, please cite this publication: 

> @inproceedings{makrigiorgis2020extracting,
>  title={Extracting the fundamental diagram from aerial footage},
>  author={Makrigiorgis, Rafael and Kolios, Panayiotis and Timotheou, Stelios and Theocharides, Theocharis and Panayiotou, Christos G},
>  booktitle={2020 IEEE 91st Vehicular Technology Conference (VTC2020-Spring)},
>  pages={1--5},
> year={2020},
>  organization={IEEE}
> }

# Package Requirements:
- OpenCV (cuda build recommended, version>=4.4 for the latest yolov4 models)
- Numpy
- scipy
- scikit-learn


In order to run the application change the config.ini file to your needs and execute 'python Monitoring.py' in the terminal
Tiny yolov3 model trained on our own vehicle dataset is provided in Data/Configs. 
The dataset images used for the training are not currently provided.
You can also change the classes/weights/configuration files in data/Configs to your own trained Yolo model (v4 is supported as well for OpenCV version >=4.4).
More accurate data results are also available at data/Extra_results/. They were exported using tiny YoloV4 trained on our latest updated dataset. Tiny YoloV4 model and the dataset used for training are currently not provided.


# Configuration File Instructions (config.ini)

    [Video]
https://user-images.githubusercontent.com/15671165/113510224-82e03f00-9562-11eb-8033-529ee1d7b0c3.mp4


https://user-images.githubusercontent.com/15671165/113510213-752ab980-9562-11eb-9d8a-43f45b566e49.mp4


    video_filename		= string	, path to the video file
    resize 			= boolean	, True if willing to resize the image before processing (improving performance)
    image_width 		= int		, Width of the resize image (if resize=True)
    image_height 		= int		, Height of the resize image (if resize=True)
    display_video		= boolean	, True for displaying result video while processing (not working on codeocean)
    display_width 		= int		, Width of the display window size(if display_video=True)
    display_height 		= int		, Height of the display window size (if display_video=True)
    
    [Export]
    save_video_results	= boolean	, True to export the result in a video
    video_export_path 	= string	, Path for the exported video to be saved (if save_video_results=True)	
    csv_path 		= string	, Path for the exported data from the video to be saved in CSV format (if export=True)
    export			= boolean	, True to export the result's video file.
    display_track		= boolean	, True to display vehicle trajectories on the display window
    
    [Detector]
    darknet_config 		= string	, Path for the darknet model cfg file
    darknet_weights 	= string	, Path for the darknet model weights file
    classes_file 		= string	, Path for the classes file
    cfg_size_width		= int		, Resize width of the image during the detection process 
    cfg_size_height		= int		, Resize height of the image during the detection process 
    iou_thresh 		= float		, IOU threshold 
    conf_thresh 		= float		, Confidence threshold 
    nms_thresh 		= float		, Non Maximum Suppression (NMS) threshold 
    use_gpu 		= boolean	, True to run the detection on Cuda GPU (works only if OpenCV is built with cuda) 
    
    [Tracker]
    flight_height 		= int		, Flight's height in meters
    sensor_height 		= float		, Camera's sensor height eg. 0.455
    sensor_width		= float		, Camera's sensor width eg. 0.617
    focal_length		= float		, Camera's focal length eg. 0.567
    reset_boxes_frames	= int		, Determine the range of frames that vehicles are set as inactive if not found.
    calc_velocity_n		= int		, Number of detections to be included in the velocity calculation
    draw_tracks		= boolean	, True to draw the trajectories on the display image / exported video
    export_data		= boolean	, True to export results of the tracking to CSV files>

For further information or questions please contact the project's website or using email at 'makrigiorgis.rafael at ucy.ac.cy'

Credits to Kios Center of Excellence: https://www.kios.ucy.ac.cy/


# Related References:

1.  YoloV4					:	https://github.com/AlexeyAB/darknet
2.  Darknet					:	https://pjreddie.com/darknet/
3.  Kios Center of Excellence	    :	https://www.kios.ucy.ac.cy/
