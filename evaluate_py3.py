# coding: utf-8
import hashlib
import urllib.request
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import pearsonr
import os
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
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

PROJECT_ROOTDIR = '/home/shuang/projects/wikimedia'
outdir='wikiImages'
root='https://upload.wikimedia.org/wikipedia/commons/thumb/'
filename_good='correctpairs.csv'
filename_bad='wrongpairs.csv'



def download_image(filename):
#this function downloads an image from the commons repository
	m = hashlib.md5()
	m.update(filename.encode('utf-8'))
	sumfilename=m.hexdigest()
	first=sumfilename[0]
	second=sumfilename[:2]
	#format is: first char of md5 sum / second char / filename / resolutions - filename
	url=root+first+'/'+second+'/'+filename+'/600px-'+filename
	urllib.request.urlretrieve(url, outdir+filename)

def extract_features(featurename,img):
    '''
    input: string of method for feature extraction
           numpy array of bgr images
    output: if the method is 'hsvHistogram' or 'rgbHistogram',
            the output would be a list of 256 long
            if the method is 'hog',
            the output would be a list
    '''
    if featurename == 'hsvHistogram':
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        color = ('b','g','r')
        for i,col in enumerate(color):
            hist = cv2.calcHist([img],[i],None,[256],[0,256])
        return hist/3.0
    elif featurename = 'rgbHistogram':
        color = ('b','g','r')
        for i,col in enumerate(color):
            hist = cv2.calcHist([img],[i],None,[256],[0,256])
        return hist/3.0
    elif featurename = 'hog':
        hist = hog.compute(img1,winStride,padding,locations)
        return hist


def evaluate_distance(imagename1,imagename2):
	#downloadImage(img1)
	#downloadImage(img2)
	filename1=outdir+imagename1
	filename2=outdir+imagename2
	#read images
	try:
		img1 = cv2.imread(filename1,0)
		img2 = cv2.imread(filename2,0)
	except:
		return None
	if img1 is None or img2 is None:
		return None
	#compute hog
	hist1 = hog.compute(img1,winStride,padding,locations)
	hist2 = hog.compute(img2,winStride,padding,locations)
	# compute distance
	# dist=cosine_similarity(hist1,hist2)
	dist=pearsonr(hist1,hist2)
	correlation=0 if np.isnan(dist[0]) else dist[0]
	return correlation

def read_file_and_compute(filename):
	distances=[]
	count=0
	with open(filename) as f:
		for line in f:
			row=line[:-1].split('\t')
			img1=row[1]
			img2=row[2]
			d=evaluate_distance(img1,img2)
			if d is not None:
				distances.append(d)
			else:
				count+=1
				print(count)
	return distances
if __name__=='__main__':
    # test the the feature extractor
    test_imgpath = os.path.join(PROJECT_ROOTDIR,'images','testImage.jpg')
    test_img = cv2.imread(test_imgpath,cv2.IMREAD_COLOR)
    features = extract_features('hsvHistogram',test_img)
    print(features.shape)
# download_image()
# distances_good=read_file_and_compute(filename_good)
# distances_bad=read_file_and_compute(filename_bad)

# print 'good'+str(np.mean(np.asarray(distances_good)))

# print 'bad'+str(np.mean(np.asarray(distances_bad)))
