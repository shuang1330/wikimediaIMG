import hashlib
import urllib
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import pearsonr
import argparse
# from matplotlib import pyplot

#HOG parameters
winSize = (32,32)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
winStride = (8,8)
padding = (8,8)
locations = ((10,20),)
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                        derivAperture,winSigma,histogramNormType,L2HysThreshold,
                        gammaCorrection,nlevels)

PROJECT_DIR = '/home/shuang/projects/wikimedia'
imageFolder=os.path.join(PROJECT_DIR,'wikiImages')
metaDataDir = os.path.join(PROJECT_DIR,'metadata')
root='https://upload.wikimedia.org/images.zipwikipedia/commons/thumb/'
filename_good=os.path.join(metaDataDir,'correctpairs.csv')
filename_bad=os.path.join(metaDataDir,'wrongpairs.csv')

def extract_features(featurename,img,
                     transform=True):
    '''
    input: string of method for feature extraction
           numpy array of bgr images
    output: if the method is 'hsvHistogram' or 'rgbHistogram',
            the output would be a list of 256 long
            if the method is 'hog',
            the output would be a list
    '''
    if transform:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[1]

    if featurename == 'hsvHistogram':
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hist = []
        for i in range(3):
            hist.append(cv2.calcHist([img],[i],None,[256],[0,256]))
        return hist[1]
    elif featurename == 'rgbHistogram':
        color = ('b','g','r')
        for i,col in enumerate(color):
            hist = cv2.calcHist([img],[i],None,[256],[0,256])
        return hist/3.0
    elif featurename == 'hog':
        hist = hog.compute(img,winStride,padding,locations)
        print(hist)
        return hist

def download_image(filename):
#this function downloads an image from the commons repository
	m = hashlib.md5()
	m.update(filename)
	sumfilename=m.hexdigest()
	first=sumfilename[0]
	second=sumfilename[:2]
	#format is: first char of md5 sum / second char / filename / resolutions - filename
	url=root+first+'/'+second+'/'+filename+'/600px-'+filename
	urllib.urlretrieve(url, outdir+filename)

def evaluate_distance(imagename1,imagename2,color_channel,featurename):
    filename1=os.path.join(imageFolder,imagename1)
    filename2=os.path.join(imageFolder,imagename2)
    try:
        if color_channel == 'gray':
            img1 = cv2.imread(filename1,0)
            img2 = cv2.imread(filename2,0)
        elif color_channel == 'color':
            img1 = cv2.imread(filename1,cv2.IMREAD_COLOR).astype(np.uint8)
            img2 = cv2.imread(filename1,cv2.IMREAD_COLOR).astype(np.uint8)
    except:
        return None
    if img1 is None or img2 is None:
        return None

    hist1 = extract_features(featurename,img1)
    hist2 = extract_features(featurename,img2)
    #

    dist=pearsonr(hist1,hist2)
    correlation=0 if np.isnan(dist[0]) else dist[0]
    return correlation

def read_file_and_compute(filename,featurename,color_channel):
	distances=[]
	count=0
	with open(filename) as f:
		for line in f:
			row=line[:-1].split('\t')
			img1=row[1]
			img2=row[2]
			d=evaluate_distance(img1,img2,color_channel,featurename)
			if d is not None:
				distances.append(d)
			else:
				count+=1
	return distances,count

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description=
    'Build overlapping ratio matrix for all Roadmap features.')
    parser.add_argument('--color', dest='color_channel',
                        help='color channel to use, either gray or color',
                        default=None, type=str)
    parser.add_argument('--feature', dest='featurename',
                        help='features to use for classifying the images, \n'+\
                             'options being hog, rgbHistogram, hsvHistogram',
                        default=None, type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:\n')
    print(args)

    distances_good,count1=read_file_and_compute(filename_good,args.featurename,args.color_channel)
    distances_bad,count2=read_file_and_compute(filename_bad,args.featurename,args.color_channel)

    print('good'+str(np.mean(np.asarray(distances_good))))

    print('bad'+str(np.mean(np.asarray(distances_bad))))
