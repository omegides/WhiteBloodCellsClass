
import matplotlib.image as im
import numpy as np
import pandas as pd
import cv2
from skimage import color, measure, filters
from os import getcwd, listdir, path
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler



def mask_an_rgb_image(rgb_img,bw_img): #function to help calculate entropy of the image for each channel (RGB)
    img_R = bw_img.ravel()*rgb_img[:,:,0].ravel()
    img_G = bw_img.ravel()*rgb_img[:,:,1].ravel()
    img_B = bw_img.ravel()*rgb_img[:,:,2].ravel()
    masked_img = np.dstack((img_R.reshape(bw_img.shape),img_G.reshape(bw_img.shape),
               img_B.reshape(bw_img.shape)))
    return masked_img
    
def entropy_calc(pdf): #entropy calculation
    pdf[pdf==0] = 1
    entropy = -(pdf*np.log2(pdf)).sum()
    return entropy

def extract_features_for_clustering(images = []):
    idx_max = []
    df = pd.DataFrame([],columns=[
                       'Area',
                       'Perimeter'
                      ,'Perimeter^2/Area'
                      ,'HISTCMP_CHISQR_RG','HISTCMP_CHISQR_RB','HISTCMP_CHISQR_BG'
                      ,'hist_calc_CHIS_GR','hist_calc_CHIS_BR','hist_calc_CHIS_GB'
                      ,'ent_gray'
                      ,'ent_R','ent_G','ent_B'])
    
    for image in images:
    ## first, I converted each image to L*a*b presentation
        im_lab = color.rgb2lab(image)
    ## I took channel 'a' and 'b' to enhance the pixels that importent in each image    
        sum_img = (im_lab[..., 2] - im_lab[..., 1])
        image_scaled = sum_img+abs(sum_img).max()
        image_scaled = image_scaled/image_scaled.max()*255
    ## Filter the image to reduce rough changes in each image (softer image)
        image_scaled_filt = filters.gaussian(image_scaled,sigma=0.4)
    ## For segmentation I took threshold (Otsu's threshold)
        image_otsu = filters.threshold_otsu(image_scaled_filt)
        segmentation = image_scaled_filt<image_otsu;
    ## Found labels to the BW image after thersholding    
        labels = measure.label(segmentation)
    ## Extract region properties
        props = measure.regionprops(labels)
    
        data = np.array([(prop.area,
                          prop.perimeter
                          ,(prop.perimeter)**2/prop.area
                          ) for prop in props])
    # Found the largest region (ignoring one if there's 2)
        idx_max = (data[:,0].argmax())
    # Taking only the region with the white blood cell    
        box_sizes = props[idx_max].bbox
        image_cropped = image[box_sizes[0]:box_sizes[2],box_sizes[1]:box_sizes[3],:]
        image_cropped_nucl = segmentation[box_sizes[0]:box_sizes[2],box_sizes[1]:box_sizes[3]]
        img_gray = np.uint8(color.rgb2gray(image_cropped)*255)
    # Image of the nuclei segmented
        masked_img = mask_an_rgb_image(image_cropped,image_cropped_nucl)
    
    # Histogram of each channel (RGB) and histogram of the gray scaled image    
        histR = cv2.calcHist(image_cropped[...,0],[0],None,[256],[0,256])
        histG = cv2.calcHist(image_cropped[...,1],[0],None,[256],[0,256])
        histB = cv2.calcHist(image_cropped[...,2],[0],None,[256],[0,256])
        histGray = cv2.calcHist([img_gray],[0],None,[256],[0,256])
        hist_for_entropy_R = cv2.calcHist(masked_img[...,0],[0],None,[256],[0,256])
        hist_for_entropy_G = cv2.calcHist(masked_img[...,1],[0],None,[256],[0,256])
        hist_for_entropy_B = cv2.calcHist(masked_img[...,2],[0],None,[256],[0,256])
    # Calculating entropy
        ent_R = np.array([entropy_calc(hist_for_entropy_R/256)])
        ent_G = np.array([entropy_calc(hist_for_entropy_G/256)])
        ent_B = np.array([entropy_calc(hist_for_entropy_B/256)])
        ent_gray = np.array([entropy_calc(histGray/256)])
    # Chi-square using cv2.compareHist
        hist_calc_CHIS_RG = np.array([cv2.compareHist(histR, histG, method=cv2.HISTCMP_CHISQR)])
        hist_calc_CHIS_RB = np.array([cv2.compareHist(histR, histB, method=cv2.HISTCMP_CHISQR)])
        hist_calc_CHIS_BG = np.array([cv2.compareHist(histB, histG, method=cv2.HISTCMP_CHISQR)])
        hist_calc_CHIS_GR = np.array([cv2.compareHist(histG, histR, method=cv2.HISTCMP_CHISQR)])
        hist_calc_CHIS_BR = np.array([cv2.compareHist(histB, histR, method=cv2.HISTCMP_CHISQR)])
        hist_calc_CHIS_GB = np.array([cv2.compareHist(histG, histB, method=cv2.HISTCMP_CHISQR)])
    # Fix all the feature in one array list    
        df.loc[-1] = np.concatenate((data[idx_max,:],hist_calc_CHIS_RG,hist_calc_CHIS_RB,hist_calc_CHIS_BG
                                     ,hist_calc_CHIS_GR,hist_calc_CHIS_BR,hist_calc_CHIS_GB
                                     ,ent_gray,ent_R,ent_G,ent_B),axis=0)
        df.index = df.index + 1
    
    df = df.sort_index()
    return df

def Train(images = []):
    model = None

    fileNames = []
    images_file = []
    for filename in images:
        img = im.imread(filename)
        fileNames.append(filename)
        if img is not None:
            images_file.append(img)
    
    
    df = extract_features_for_clustering(images_file)
    scaler = MinMaxScaler();
    norm_dataframe = scaler.fit_transform(df)
    model = KMeans(init='random',n_clusters=4, random_state=0).fit(norm_dataframe)
    
    raw_data = {'File Name' :fileNames,'Labels':model.labels_}
    df2csvfile = pd.DataFrame(data=raw_data,columns=['File Name' ,'Labels'])
    df2csvfile.to_csv(getcwd()+'\Labeld_Names_trained.csv')
    return model

def Test(model,images = []):
    cluster_ids = []

    fileNames = []
    images_file = []
    for filename in images:
        img = im.imread(filename)
        fileNames.append(filename)
        if img is not None:
            images_file.append(img)
            
    df = extract_features_for_clustering(images_file)
    model.fit_predict(df)
    cluster_ids = model.labels_    
    raw_data = {'File Name' :fileNames,'Labels':model.labels_}
    df2csvfile = pd.DataFrame(data=raw_data,columns=['File Name' ,'Labels'])
    df2csvfile.to_csv(getcwd()+'\Labeld_Names_tested.csv')
    return cluster_ids

def load_images_list_from_folder(folder): #load a batch of images and create a list, to check the code
    images = []
    for filename in listdir(folder):
        images.append(path.join(folder,filename))
    return images
#%% checking code section
''' load_images_list_from_folder function input is the name of the folder the images are in for each of the casses. 
    and than the Train and Test functions getting a list of (path,image_name) and reading with it the images  
''' 
images_list_train = load_images_list_from_folder(r'D:\M Sc BME\Machine learning course\HW01\Train')
kmeans_model = Train(images_list_train) 

images_list_test = load_images_list_from_folder(r'D:\M Sc BME\Machine learning course\HW01\Test')
Test(model=kmeans_model,images=images_list_test)
