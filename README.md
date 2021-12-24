Soy disease detection

# 1 - problem

The proposed demonstration for the detection of diseases in soybean will benefit from one of the Deep Learning(DL) techniques in the field of Computer Vision or Computer Vision(CV). This being the Convolutional Neural Networks or CNNs which, in turn, have the ability to learn (train) representations from a set of image samples using computational models with several layers. 
  This methodology approached makes use of an image collection that was submitted to a data augmentation process to increase the quantity of the image set. Thus, the set of images was used for training a CNN model. 
  ### 1.2 - DISEASES
  ![Alt text](image_to_readme/disease.png?raw=true "disease")
  
  ### 1.3 - AMOUNT OF DISEASES
  ![Alt text](image_to_readme/total_images.png?raw=true "total_images")
  
  ### 2 - SAMPLE RESULTS
  Due to the limitation of computational resources, it was decided to reduce the resolution of the images to 64x64 and 128x128. Furthermore, a batch size of size 32 was used for both resolutions. Each experiment was carried out from 40 to 100 times (training). The time that reached the highest accuracy was selected, and the algorithm stops its execution when it can no longer learn/train. Finally, the architecture found was submitted to 5 (five) training and testing rounds, with the training, validation and testing set respectively. These 5 runs of experiments contribute to an investigation of the stability of the model and thus guarantee its reliability. Results below. 
  
  ### 2.1 - SAMPLE RESULTS 128X128
  ![Alt text](image_to_readme/result_128.png?raw=true "result_128.png")
  
  ### 2.2 - SAMPLE RESULTS 64X64
  ![Alt text](image_to_readme/result_64.png?raw=true "result_64.png")
  
  ### 2.3 - ACCURACY OF RESULTS 128X128
  ![Alt text](image_to_readme/ACCYRACYS128X128.png?raw=true "ACCYRACYS128X128.png")
  
   ### 2.3 - ACCURACY OF RESULTS 128X128
  ![Alt text](image_to_readme/ACCURACYS64X64.png?raw=true "ACCURACYS64X64")
