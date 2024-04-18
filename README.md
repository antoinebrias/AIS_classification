# AIS_classification
Details can be found on my website (https://www.briaslab.fr/blog/?action=view&url=vessel-classification-using-ais-data).

This project aims at classifying vessel categories based on their trajectory. This repo contains 3 scripts:
  - streamlit_vessel.py to vizualize data
  - trajectory_pixelization.py to build images based on the dataset trajectories
  - cnn_vessel.py t build a CNN trained to classify vessels categories

Warning : since the available dataset is small, the CNN may not perform very well. However the workflow should be correct. 
