# e6691-2022spring-assign2-SURG-an3078-bmh2168-wab2138



## Assignment 2 - Surgical Phase Recognition
### E6691 Spring 2022

#### About
Assignment 2 for E6691 Spring 2022. In this assignment we attempt to detect phases of hernia surgeries through videos.

The ./utils folder contains relevant preprocessing functions to convert videos to images, and the implementation of the Dataset classes for the training of the different models.
#### Saved Model Weights
* https://drive.google.com/drive/folders/1kmv8pb2Zxfp-Fe-ZMMG4lW9ZxgYFHMqo?usp=sharing

#### Project Structure
* Setup.ipynb
* preprocessing.ipynb <strong><em>Preprocessing the training labels</em></strong>
* EfficientNet.ipynb <strong><em>Efficientnet Training</em></strong>
* Resnet18_training_result.ipynb <strong><em>ResNet Training</em></strong>
* LSTM_sequence_cleaning.ipynb <strong><em>Sequence to sequence model for error correction</em></strong>
* Resnet-LSTM.ipynb <strong><em>CNN-LSTM implementation</em></strong>
* Predictions.ipynb <strong><em>Code that predicts on the test data with the CNN and LSTM error correcting models</em></strong>
* video.phase.trainingData.clean.StudentVersion.csv <strong><em>Cleaned labels for the training data</em></strong>
* all_labels_hernia.csv <strong><em>List of the classes</em></strong>
* my_kaggle_preds.csv <strong><em>Test set predictions</em></strong>
* kaggle_template.csv <strong><em>Template for the test prediction output</em></strong>
* README.md
* utils/
  * ImagesDataset.py <strong><em>Dataset class for the ResNet and EfficientNet models</em></strong>
  * SequenceDataset.py  <strong><em>Dataset class for the error correcting LSTM network</em></strong>
  * VideosDataset.py <strong><em>Dataset class for the CNN-LSTM model</em></strong>
  * image_extraction.py <strong><em>Code to extract jpeg images from videos</em></strong>
  * workers.py <strong><em>Code to extract jpeg images from videos</em></strong>

**References**
* Rémi Cadène, Thomas Robert, Nicolas Thome, Matthieu Cord. “M2CAI Workflow Challenge: Convolutional Neural Networks with Time Smoothing and Hidden Markov Model for Video Frames Classification” Sorbonne Universites, UPMC Univ Paris 06, CNRS, LIP6 UMR 7606 (2016). 
  * https://arxiv.org/abs/1610.05541
* Aksamentov I, Twinanda AP, Mutter D, et al. “Deep Neural Networks Predict Remaining Surgery Duration from cholecystectomy videos.” 2017 International Conference on Medical Image Computing and Computer-Assisted Intervention: Springer; (2017): 586–593. 
  * https://link-springer-com.ezproxy.cul.columbia.edu/chapter/10.1007/978-3-319-66185-8_66 
