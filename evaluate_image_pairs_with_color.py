import hashlib
import urllib
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import pearsonr
from matplotlib import pyplot

#HOG parameters
winSize = (32,32)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins =9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 32
winStride = (8,8)
padding = (8,8)
locations = ((10,20),)
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#compute(img[, winStride[, padding[, locations]]]) -> descriptors

# time it

outdir='images/'
root='https://upload.wikimedia.org/wikipedia/commons/thumb/'
filename_good='correctpairs.csv'
filename_bad='wrongpairs.csv'



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
	#compute distance
	#dist=cosine_similarity(hist1,hist2)
	dist=pearsonr(hist1,hist2)
	correlation=0 if np.isnan(dist[0]) else dist[0]
	return correlation

def evaluate_color_distance(imagename1,imagename2):
	filename1=outdir+imagename1
	filename2=outdir+imagename2
	#read images
	try:
		img1 = cv2.imread(filename1,cv2.IMREAD_COLOR).astype(np.uint8)
		img2 = cv2.imread(filename2,cv2.IMREAD_COLOR).astype(np.uint8)
		#img1 = cv2.imread(filename1,0)
		#img2 = cv2.imread(filename2,0)
	except:
		return None
	if img1 is None or img2 is None:
		return None
	#compute hog
	#color = ('b','g','r')
	dist=0
	for i in range(len(img1.shape)):
		hist1 = cv2.calcHist([img1],[i],None,[256],[0,256])
		hist2 = cv2.calcHist([img2],[i],None,[256],[0,256])
		dist+=cosine_similarity(hist1.T,hist2.T)
	#compute distance
	#dist=cosine_similarity(hist1,hist2)
	#correlation=0 if np.isnan(dist) else dist/float(3)
	correlation=0 if np.isnan(dist) else dist
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
			dc=evaluate_color_distance(img1,img2)
			if d is not None:
				distances.append(d+dc)
			else:
				count+=1
				print count
	return distances

distances_good=read_file_and_compute(filename_good)
distances_bad=read_file_and_compute(filename_bad)

print 'good'+str(np.mean(np.asarray(distances_good)))

print 'bad'+str(np.mean(np.asarray(distances_bad)))




frq1,edge1=np.histogram(distances_good,range=[-2,2])
frq2,edge2=np.histogram(distances_bad,range=[-2,2])

pyplot.bar(edge1[:-1], frq1, width=np.diff(edge1), ec="k", align="edge", alpha=0.5, label='good')
pyplot.bar(edge2[:-1], frq2, width=np.diff(edge2), ec="k", align="edge", alpha=0.5, label='bad')

#pyplot.bar([0],hist_good[0], alpha=0.5, label='good',width=2)
#pyplot.bar([0],hist_bad[0], alpha=0.5, label='bad',width=2)
pyplot.legend()
pyplot.show()
