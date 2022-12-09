# Seeing-AI-system
A system of recognizing masked people's face. 

### Files with specifications
#### Appendix Folder
This contains three supporting files for preprocessing procedure.


Our cropped and masked-face dataset, augmented training set, and their corresponding label csv files have been uploaded to Google Drive, the link to the Google Drive: \url{ https://drive.google.com/drive/folders/1GvQS-xWG699w6QhPKdgc0-7fSdQWxBCo?usp=share_link}.


The original LFW can be downloaded from Kaggle.

#### data folder
The data folder contains the LFW dataset and the generated masked dataset, which would not be uploaded on the github.

#### saved models dataset
The savedmodels folder contains 4 saved models for the 4 fine-tuned dataset. Model_parameter.pth represents the best model saved for InceptionResNet v1, Model_parameter0.pth represents the best model saved for data-argumentation of InceptionResNet v1, model_vgg_parameter.pth represents the best model saved for VGG16, and model_parameter_ResNet50 saved for the best model for ResNet50. 

#### Train, test, val
The Train, test, val dataset is for each NN to train. The pictures inside are with mask faces which are already cropped. 

The above three datasets are too big to upload, which can be seen in the link.

### ipynb files and python functions
BadPredictionCheck.ipynb is to check the wrongly classified pictures \\

CreateCropFace.ipynb is to generate the train, test, val three dataset from the original LFW face dataset.\\

Cropped Upper Half Rec.ipynb\\

DatasetGeneration.ipynb is the procedure for selecting 50 categories which is suitable for our masked face recognition model. \\

Eigenface.ipynb contains the procedure of generating eigenfaces and using eigenface method for face recognition.\\

TrainwithInceptionResnet.ipynb fine tunes InceptionResnet v1 using the wearing-after-cropping faces for training.\\

facecrop_train_dlib.ipynb uses the cropped faces with dlib DNN detection and get a poor result.\\

FinetuneResnet50.ipynb fine tunes Resnet50 using the wearing-after-cropping faces for training.\\

FinetuneVGG16.ipynb fine tunes VGG16 using the wearing-after-cropping faces for trianig.\\

TrainWithDataAugmentation.ipynb applies data augmentation on our original dataset and uses InceptionResnet v1 for training.\\

generate_nontarget_data.ipynb is to generate the faces in the categories off the chosen faces, but in the whole LFW dataset for our exponential margin theory.\\




