# WaterSCNet
WaterSCNet is a deep learning model for segmentation and connectivity reconstruction of urban rivers from Sentinel-2 multi-spectral imagery. 

WaterSCNet consists of two cascaded subnetworks called WaterSCNet-segmentation (WaterSCNet-s) and WaterSCNet-connection (WaterSCNet-c).

Instructions for use
-------
        git clone git@github.com:ZXDui/WaterSCNet.git
        cd WaterSCNet

Datatset preparation
-------
Firstly, you should generate experimental sample data for training and evaluation. Here is the example:
1. Convert Sentinel-2 images downloaded from Google Earth Engine from Tiff to HDR:

        python gee_sen_ndwi.py

2. Cut the original Sentinel-2 remote sensing data and corresponding segmentation labels and connectivity labels:

        python data_crop.py

3. Divide the experimental dataset and obtain a reading list of the experimental dataset:

        python train_test_split.py
        python train_test_list.py
        

Model training
-------
You can use the following command to train WaterSCNet-s and WaterSCNet-v:

    python train_waterscnet.py

After training is done, you can use the following command to obtain the results of river segmentation and connectivity reconstruction:

    python pred_waterscnet.py

Sample data
-------
You can find sample data in a folder called "data", which includes remote sensing data, segmentation labels, connectivity labels, and RGB images for visual visualization.
