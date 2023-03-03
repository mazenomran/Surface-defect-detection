import cv2
import pandas as pd
import numpy as np
import os
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
import time
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
import pickle
from sklearn.ensemble import RandomForestClassifier

train_path = "D:/Tile Defects Detection/Multiclass data/train/"
test_path = "D:/Tile Defects Detection/Multiclass data/test/"
mask_path ="D:/Tile Defects Detection/Multiclass data/masks/"

#Features extraction function
def feature_extractor(path,mask_path):
    Dataset = pd.DataFrame()
    for image in os.listdir(path):  # iterate through each file
        #print(image)
        df = pd.DataFrame()
        if image.split('.')[0][:3] in ["cra","glu","gra","oil","rou"]: # first three letters of each defect type
            mask = cv2.imread(mask_path + image)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            bi_mask = cv2.threshold(mask,127,255,0)
            bi_mask = cv2.resize(mask, (210, 210))
            df["label"] = bi_mask.reshape(-1) # the label of each defected tile images is its binary mask
        else:
            df["label"] = np.zeros((210, 210)).reshape(-1) #if the image not for defected tile its label is (0)
               
        input_img = cv2.imread(path + image)  # Read images
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (210, 210))
        
        
        pixel_values = img.reshape(-1)/255.0
        df['Pixel_Value'] = pd.DataFrame(pixel_values)   #Pixel value itself as a feature
        #Defining the desired filter (feature).    
        edge_roberts = roberts(img)
        edge_roberts1 = edge_roberts.reshape(-1)
        df['Roberts'] = pd.DataFrame(edge_roberts1)

        # SOBEL
        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df['Sobel'] = pd.DataFrame(edge_sobel1)

        # VARIANCE with size=3
        variance_img = nd.generic_filter(img, np.var, size=3)
        edge_variance = variance_img.reshape(-1)/255.0
        df['variance'] = pd.DataFrame(edge_variance)

        # GAUSSIAN with sigma=3
        gaussian_img = nd.gaussian_filter(img, sigma=3)
        gaussian_img1 = gaussian_img.reshape(-1)/255.0
        df['Gaussian3'] = pd.DataFrame(gaussian_img1)
            
        # SCHARR
        edge_scharr = scharr(img)
        edge_scharr1 = edge_scharr.reshape(-1)
        df['Scharr'] = pd.DataFrame(edge_scharr1)

        # PREWITT
        edge_prewitt = prewitt(img)
        edge_prewitt1 = edge_prewitt.reshape(-1)
        df['Prewitt'] = pd.DataFrame(edge_prewitt1)
        # MEDIAN with sigma=3
        median_img = nd.median_filter(img, size=3)
        median_img1 = median_img.reshape(-1)/255.0
        df['Median3'] = pd.DataFrame(median_img1)
           
        # CANNY EDGE
        edges = cv2.Canny(img, 100, 200)  # Image, min and max values
        edges1 = edges.reshape(-1)/255.0
        df['Canny_Edge'] = pd.DataFrame(edges1)
        
        #Add column to original dataframe
        
        Dataset = Dataset.append(df)
    
    return Dataset
  
#Preparing training set  
Training_data= feature_extractor(train_path,mask_path)   
X_train= Training_data.drop(labels =['label'], axis=1) 
X_train.info()
y_train = Training_data['label'].values

#Training the model
RF_model = RandomForestClassifier() 
#Training time calculation
t0 = time.time()
RF_model.fit(X_train,y_train)
Training_time = time.time()-t0
print("Training_time", Training_time)

#Estimating each feature importance    
for score, name in zip(RF_model.feature_importances_, X_train.columns):
    print(round(score, 2), name)
    
#Preparing testing set     
Testing_data = feature_extractor(test_path,mask_path)
X_test = Testing_data.drop(labels =['label'], axis=1) 
y_test = Testing_data['label'].values
#Calculating model accuracy
test_prediction = RF_model.predict(X_test)   
print ("Accuracy RF_model = ", metrics.accuracy_score(y_test, test_prediction))

  

      
    