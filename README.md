# Object Detection in an Urban Environment
Repository to detect objects in an Urban Environment using Tensorflow Object Detection API. 

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `test` folder contains frames from the 10 fps video without downsampling.

You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.

### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/training_and_validation/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission Template

### Project overview
The overall objective of the project is to create a convolutional neural network to detect and classify objects namely vehicles, cyclists and pedestrians using a dataset of images of urban environments.Waymo opens dataset is used to train convolution neural network, the dataset contains annotated images of vehicles, pedestrains and cyclists.

The model training data can be downloaded directly from [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/). Preprocessed data was already provided in workspace under data/waymo/training_and_validation. Test data was also provided under data/waymo/test.

The project can be divided into three stages:
    1. Dataset Analysis and Dataset split
        - Explore and visualize the provided data set images with annotation bounding boxes
        - Analyze attributes of the dataset images including locations, light intensities, contrast etc
        - Analyze class distribution frequencies in dataset, explore possible image augmentation to remidiate
          the deficiencies of the dataset.
    2. Model Training
        - Split data into training and validation based on Data Analysis
        - Download pretrained Single Shot Detector Resnet 50 model
        - Train and validate model, visualize loss metrics for both datasets via tensorboard
    3. Model Improvement
        - Adjust learning rates, add learning rate annealing
        - Add image augmentations to improve performance of validation dataset and avoid overfitting

### Set up
Udacity Virtual Machine was utilized to complete this project, no setup was required as VM has all necessary dependencies and extensions were already installed. 

### Dataset
#### Dataset analysis
The Waymo Open Dataset used to train the model contains 97 tfrecord files. Each tfrecord file contains 1 frame per 10 seconds from a 10 fps video, essentiatly each tfrecord contains images to make a 1fps video. The images are annotated bounding boxes for vehicles, pedestrians and cyclists. 

The images within the tfrecord file have distinct attributes including light condition (sunny/overcast), time of day (day/night/dusk) locations(residential/highways/countryside), weather(rainy/foggy/overcast) and density of tracked classes(high/medium/low). Following are a few examples of the images within the tfrecord files.

#### Good light conditions
![](/images/sunny_fl.png)
![](/images/hyde_out_sunny.png)

#### Distinct weather and light conditions
![](/images/foggy_mi.png) 
![](/images/rainy_mi.png)
![](/images/overcast_mi.png)

#### Different tracked class densities
![](/images/busy_st.png)
![](/images/low_den.png)

#### Data Distribution Analysis
The Waymo Open Dataset used to train the Single Shot Detector Model is heavily biased to detect Vehicles and Pedestrians. Based on an analysis of 1000 shuffled images available in the training dataset, vehicles accounted for 77% of the tracked classes followed by pedestrian at 23% and cyclists at 1%.

![](/images/class_distribution.png =150x)

The percentage of tracked cyclist in 20000 random images reduces to 0.56%, with tracked pedestrians at 22.41% and vehicles at 77.03%. 

![](/images/class_distribution_2.png =150x)

Training an Object Detection Model on dataset with uneven distribution of tracked classes doesnt not bode well for the tracked class with lowest frequency. In this case the model wont predict or track cyclists with great accuracy and can mislable tracked cyclists.

#### Cross validation
The 97 available training and validation files were divided 85:15 split. This allows for 82 tfrecords files for training and 15 tfrecord files validation. The create_splits files first shuffles the tfrecord files before splitting the dataset. In this case shuffling allows for the dataset to be less variant, keeping the dataset representative of everyday urban driving environment and avoid overfitting.

Testing files are already provided, so the 82 tfrecord files to train a pretrained model are sufficient. The 15 tfrecord files shall be more that sufficient to exhaustively validate the trained model and help avoid overfitting.

### Training
#### Reference experiment
The Model uses a pretraing Single Shot Detector (SSD) Resnet 50 neural network model. The SSD Resnet 50 model has two image augmentations random horizontal flip and random image crop. The following shows the training and validation loss for the reference model with no additional augmentations, a learning rate of 0.04. The learning rate is annealed using a cosine decay function.

![](/images/Reference_loss.png)

As evidence by the loss metrics, this model performs poorly on the training and validation dataset with substantial loss/total_loss. The training loss is show in orange and validation loss is shown in blue. The model seems to be either stuck in a local minima based on the gradual plateau of the loss metrics.

#### Improve on the reference
Possible ways to improve the model are:
    1. Add Image augmentations, to provide greater training variety to the model
    2. Adjust or decrease learning rate
    3. Increase number of training steps

#### Image Augmentations
Following Image Augmentations were implemented in the model improvements

Brightness adjusted by a delta of 0.3. Most images seem to have perfect light condition, increasing the brightness creates overexposed images making it harder to detect features.

![](/images/bright_1.png) ![](/images/bright_2.png)

Contrast adjusted between min delta = 0.7 and max delta = 1.1. This randomly scales the image contrast between 0.7 - 1.1, emulating harsh sunlight condition creating drastic shadows and light areas.

![](/images/contrast_1.png) ![](/images/contrast_2.png)

Additional random RGB to Gray augmentation with a proability of 0.3. This negates the ability of the model to rely on color changes as distincting feature.

![](/images/gray_bright_1.png) ![](/images/gray_contrast_1.png)

#### Adjust Learning Rate
Following multiple training iteration the final training rate was set at 0.0001819 that is annealed using a consine decay.

The following are the loss metrics for experiment_2, the pipeline config file for this iteration can be found under experiments/experiment_2.

![](/images/exp_2_loss.png)

Since the density of tracked vehicles is dominant in the dataset, the model performs the best in detecting larger objects i.e. vehicles.This can be evidenced by the fact that the mean average percision for large boxes in the figure below is 0.5 and reduces to approximately 0.2 for small tracked objects.

![](/images/exp_2_map.png)

The augmented model with adjusted learning rate as depicted above performs considerable better than the reference model as evidenced by the loss metrics. The model is better at tracking vehicles in variety of representative light, weather conditions, traffic etc. Because the dataset is not balanced and does not provide considerable amounts of datapoints for cyclists and pedestrians it under performs in detecting these two tracked classes. With a better and balanced dataset, the model can be trained to perform better on all tracked classes.