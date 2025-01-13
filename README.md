# Hand-Sign---to---Text-Converter
These scripts creates a ML model with the help of opencv and mediapipe libraries.
For training process , Random Forest Classifier is used. Yes you guessed it right Classification is used to predict a meaning of hand gesture.
This project isn't created from scratch. I've learnt it from https://www.youtube.com/@ComputerVisionEngineer.
Here is the link of the video https://www.youtube.com/watch?v=MJCSjXepaAM&t=2865s
But I've updated the code in the preprocessing part by adding the process of Normalisation and Flattening.
Also added a code for inhomogeneity error. You''ll get this error from original Code.
Limitations:Not suitable for a moving gesture.Again this isn't a deep learning project.For that you may have to use a lstm model.This code is only for 42 features(x,y) , so it is only capable of showing meaning of single hand gesture.
The methodology of model creation is in process.txt file(Here you'll find detailed explaination of each script).

