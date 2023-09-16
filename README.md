# WaterSCNet
WaterSCNet is a deep learning model for segmentation and connectivity reconstruction of urban rivers from Sentinel-2 multi-spectral imagery. 

WaterSCNet consists of two cascaded subnetworks called WaterSCNet-segmentation (WaterSCNet-s) and WaterSCNet-connection (WaterSCNet-c).

You can use the following command to train WaterSCNet-s and WaterSCNet-v.

    python train_waterscnet.py

After training is done, you can use the following command to obtain the results of river segmentation and connectivity reconstruction:

    python pred_waterscnet.py

