### Imports ###
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from src.utils import *
import src.utils as hf 

### Import data ###
dirIn = "data/"
multiIm, annotationIm = hf.loadMulti('multispectral_day01.mat' , 'annotation_day01.png', dirIn)
imRGB = imread(dirIn + 'color_day01.png')
plt.imshow(imRGB)
plt.show()
plt.imshow(multiIm[:, :, 6])
plt.show()
plt.imshow(annotationIm[:, :, 0])
plt.title("annotation image, salami")
plt.show()
plt.imshow(annotationIm[:, :, 1])
plt.title("annotation image, fat pixels")
plt.show()
plt.imshow(annotationIm[:, :, 2])
plt.title("annotation image, meat pixels")
plt.show()


### Analysis of data ###
[fatPix, fatR, fatC] = hf.getPix(multiIm, annotationIm[:,:,1])
[meatPix, meatR, meatC] = hf.getPix(multiIm, annotationIm[:,:,2])
meatMean = np.mean(meatPix, 0)
fatMean = np.mean(fatPix, 0)

# Here we plot the mean values for pixels with meat and fat respectively
plt.plot(meatMean,'b', label = "meat means")
plt.plot(fatMean,'r', label = "fat mean")
plt.xticks(range(0, 19), range(0,19))
plt.xlabel("band")
plt.ylabel("pixel value")
plt.legend()
plt.show()

### Thresholding ###
day = 1
t = (meatMean + fatMean) / 2
print(t)

correctMeat = meatPix < t
wrongMeat = meatPix > t
correctFat = fatPix > t
wrongFat = fatPix < t
print("meat correctly classified: ", np.sum(correctMeat, 0))
print("meat wrongly classified: ", np.sum(wrongMeat, 0))
print("fat correctly classified: ", np.sum(correctFat, 0))
print("fat wrongly classified: ", np.sum(wrongFat, 0))

errorRate = (np.sum(wrongMeat, 0) + np.sum(wrongFat, 0)) / (np.sum(correctMeat, 0) + np.sum(wrongMeat, 0) + np.sum(correctFat, 0) + np.sum(wrongFat, 0))
print("error rate:", errorRate)
thresholdUse = np.argmin(errorRate)
threshold = t[thresholdUse]
print("best band:", thresholdUse, "with error rate:", errorRate[thresholdUse], "and threshold:", threshold)

# Classify where the salami is in the image using the best band and threshold and plot it on the color image
meatClass = multiIm[:, :, thresholdUse] < threshold
plt.imshow(imRGB)
plt.imshow(meatClass, alpha = 0.4)
plt.title("Meat classification using band " + str(thresholdUse) + " with threshold " + str(round(threshold)) + "\nYellow is meat, purple is fat")
plt.show()
plt.imshow(imRGB)
plt.show()


### Classification by all bands ###
meatCov = np.cov(meatPix, rowvar=False)
fatCov = np.cov(fatPix, rowvar=False)
meatM = meatPix.shape[0]
fatM = fatPix.shape[0]
pooledCov = (meatM * meatCov + fatM * fatCov) / (meatM + fatM)

def discriminant(x, mean, cov):
    d = x @ np.linalg.inv(cov) @ mean - 0.5 * mean.T @ np.linalg.inv(cov) @ mean
    return d

# calculate error rate using LDA in annotation image
# 1 is fat, 2 is meat
correctMeat = 0
wrongMeat = 0
correctFat = 0
wrongFat = 0
[val, r, c] = hf.getPix(multiIm, annotationIm[:,:,1])
for i in range(val.shape[0]):
    dMeat = discriminant(val[i], meatMean, pooledCov)
    dFat = discriminant(val[i], fatMean, pooledCov)
    if dMeat > dFat:
        wrongFat += 1
    else:
        correctFat += 1

[val, r, c] = hf.getPix(multiIm, annotationIm[:,:,2])
for i in range(val.shape[0]):
    dMeat = discriminant(val[i], meatMean, pooledCov)
    dFat = discriminant(val[i], fatMean, pooledCov)
    if dMeat > dFat:
        correctMeat += 1
    else:
        wrongMeat += 1

errorRateLDA = (wrongMeat + wrongFat) / (correctMeat + wrongMeat + correctFat + wrongFat)
print("error rate LDA:", errorRateLDA)

meatDiscriminant = np.array([discriminant(x, meatMean, pooledCov) for x in multiIm.reshape(-1, 19)])
fatDiscriminant = np.array([discriminant(x, fatMean, pooledCov) for x in multiIm.reshape(-1, 19)])
classification = meatDiscriminant > fatDiscriminant
classification = classification.reshape(multiIm.shape[0], multiIm.shape[1])

plt.imshow(imRGB)
plt.imshow(classification, alpha = 0.4)
plt.title("Meat classification using LDA\nYellow is meat, purple is fat")
plt.show()


### Calculations for all days ###
allImgs = [
    hf.loadMulti('multispectral_day01.mat' , 'annotation_day01.png', dirIn),
    hf.loadMulti('multispectral_day06.mat' , 'annotation_day06.png', dirIn),
    hf.loadMulti('multispectral_day13.mat' , 'annotation_day13.png', dirIn),
    hf.loadMulti('multispectral_day20.mat' , 'annotation_day20.png', dirIn),
    hf.loadMulti('multispectral_day28.mat' , 'annotation_day28.png', dirIn),
]
days = ["01", "06", "13", "20", "28"]

## Model 1 ## 
# error rates for remaining days using thresholding
errorRates = []
for mIm, aIm in allImgs:
    [fatPix, fatR, fatC] = hf.getPix(mIm, aIm[:,:,1])
    [meatPix, meatR, meatC] = hf.getPix(mIm, aIm[:,:,2])
    correctMeat = meatPix < threshold
    wrongMeat = meatPix > threshold
    correctFat = fatPix > threshold
    wrongFat = fatPix < threshold
    errorRate = (np.sum(wrongMeat, 0) + np.sum(wrongFat, 0)) / (np.sum(correctMeat, 0) + np.sum(wrongMeat, 0) + np.sum(correctFat, 0) + np.sum(wrongFat, 0))
    errorRates.append(errorRate[thresholdUse])

for i, e in enumerate(errorRates):
    print("error rate thresholding for day " + days[i] + ": " + str(e))

# classify the rest of the images using thresholding and plot the results
for i, (mIm, aIm) in enumerate(allImgs):
    imRGB = imread(dirIn + 'color_day' + days[i] + '.png')
    meatClass = mIm[:, :, thresholdUse] < threshold
    plt.imshow(imRGB)
    plt.imshow(meatClass, alpha = 0.4)
    plt.title("Meat classification using band " + str(thresholdUse) + " with threshold " + str(round(threshold)) + "\nYellow is meat, purple is fat")
    plt.show()

## Model 2 ##
# error rates for remaining days
errorRatesLDA = []
for mIm, aIm in allImgs:
    [val, r, c] = hf.getPix(mIm, aIm[:,:,1])
    correctMeat = 0
    wrongMeat = 0
    correctFat = 0
    wrongFat = 0
    for i in range(val.shape[0]):
        dMeat = discriminant(val[i], meatMean, pooledCov)
        dFat = discriminant(val[i], fatMean, pooledCov)
        if dMeat > dFat:
            wrongMeat += 1
        else:
            correctFat += 1

    [val, r, c] = hf.getPix(mIm, aIm[:,:,2])
    for i in range(val.shape[0]):
        dMeat = discriminant(val[i], meatMean, pooledCov)
        dFat = discriminant(val[i], fatMean, pooledCov)
        if dMeat > dFat:
            correctMeat += 1
        else:
            wrongFat += 1

    errorRateLDA = (wrongMeat + wrongFat) / (correctMeat + wrongMeat + correctFat + wrongFat)
    errorRatesLDA.append(errorRateLDA)

for i, e in enumerate(errorRatesLDA):
    print("error rate LDA for day " + days[i] + ": " + str(e))

#Classify the rest of the images using LDA and save the results in a list for further use (e.g. calculating error rate, plotting etc.)
classifications = []
for mIm, aIm in allImgs:
    meatDiscriminant = np.array([discriminant(x, meatMean, pooledCov) for x in mIm.reshape(-1, 19)])
    fatDiscriminant = np.array([discriminant(x, fatMean, pooledCov) for x in mIm.reshape(-1, 19)])
    classification = meatDiscriminant > fatDiscriminant
    classification = classification.reshape(mIm.shape[0], mIm.shape[1])
    classifications.append(classification)

# Show all classifications next to the color images
for i, (mIm, aIm) in enumerate(allImgs):
    imRGB = imread(dirIn + 'color_day' + days[i] + '.png')
    plt.imshow(imRGB)
    plt.imshow(classifications[i], alpha = 0.4)
    plt.title("Meat classification using LDA for day " + days[i] + "\nYellow is meat, purple is fat")
    plt.show()


### Train LDA on each day and test on the other days ###
def LDA(mMean, fMean, cov):
    # error rates for remaining days
    errorRatesLDA = []
    for mIm, aIm in allImgs:
        [val, r, c] = hf.getPix(mIm, aIm[:,:,1])
        correctMeat = 0
        wrongMeat = 0
        correctFat = 0
        wrongFat = 0
        for i in range(val.shape[0]):
            dMeat = discriminant(val[i], mMean, cov)
            dFat = discriminant(val[i], fMean, cov)
            if dMeat > dFat:
                wrongMeat += 1
            else:
                correctFat += 1

        [val, r, c] = hf.getPix(mIm, aIm[:,:,2])
        for i in range(val.shape[0]):
            dMeat = discriminant(val[i], mMean, cov)
            dFat = discriminant(val[i], fMean, cov)
            if dMeat > dFat:
                correctMeat += 1
            else:
                wrongFat += 1

        errorRateLDA = (wrongMeat + wrongFat) / (correctMeat + wrongMeat + correctFat + wrongFat)
        errorRatesLDA.append(errorRateLDA)

        # for i, e in enumerate(errorRatesLDA):
        #     print("error rate LDA for day " + days[i] + ": " + str(e))

        # #Classify the rest of the images using LDA and save the results in a list for further use (e.g. calculating error rate, plotting etc.)
        # classifications = []
        # for mIm, aIm in allImgs:
        #     meatDiscriminant = np.array([discriminant(x, meatMean, cov) for x in mIm.reshape(-1, 19)])
        #     fatDiscriminant = np.array([discriminant(x, fatMean, cov) for x in mIm.reshape(-1, 19)])
        #     classification = meatDiscriminant > fatDiscriminant
        #     classification = classification.reshape(mIm.shape[0], mIm.shape[1])
        #     classifications.append(classification)
        # # Show all classifications next to the color images
        # for i, (mIm, aIm) in enumerate(allImgs):
        #     imRGB = imread(dirIn + 'color_day' + days[i] + '.png')
        #     plt.imshow(imRGB)
        #     plt.imshow(classifications[i], alpha = 0.4)
        #     plt.title("Meat classification using LDA for day " + days[i] + "\nYellow is meat, purple is fat")
        #     plt.show()

    return errorRatesLDA

# get error rates for LDA using the means and covariances calculated from all days
errorRatesLDA = []
for mIm, aIm in allImgs:
    [fPix, fR, fC] = hf.getPix(mIm, aIm[:,:,1])
    [mPix, mR, mC] = hf.getPix(mIm, aIm[:,:,2])
    mMean = np.mean(mPix, 0)
    fMean = np.mean(fPix, 0)
    mCov = np.cov(mPix, rowvar=False)
    fCov = np.cov(fPix, rowvar=False)
    meatM = mPix.shape[0]
    fatM = fPix.shape[0]
    pooledCov = (meatM * mCov + fatM * fCov) / (meatM + fatM)
    errorRatesLDA.append(LDA(mMean, fMean, pooledCov))

for i in range(5):
    for j in range(5):
        print("error rate using day:", days[i], "on day", days[j], "is", errorRatesLDA[i][j])
    print()

# tabular form
for i in range(5):
    for j in range(5):
        if i == j:
            print("X", end="\t")
        else:
            print(round(errorRatesLDA[i][j], 5), end="\t")
    print()


### Prior knowledege ###
pFat = 0.3
pMeat = 0.7

def discriminantPrior(x, mean, cov, prior):
    d = x @ np.linalg.inv(cov) @ mean - 0.5 * mean.T @ np.linalg.inv(cov) @ mean + np.log(prior)
    return d

def LDAPrior(mMean, fMean, cov, priorMeat, priorFat):
    # error rates for remaining days
    errorRatesLDA = []
    for mIm, aIm in allImgs:
        [val, r, c] = hf.getPix(mIm, aIm[:,:,1])
        correctMeat = 0
        wrongMeat = 0
        correctFat = 0
        wrongFat = 0
        for i in range(val.shape[0]):
            dMeat = discriminantPrior(val[i], mMean, cov, priorMeat)
            dFat = discriminantPrior(val[i], fMean, cov, priorFat)
            if dMeat > dFat:
                wrongMeat += 1
            else:
                correctFat += 1

        [val, r, c] = hf.getPix(mIm, aIm[:,:,2])
        for i in range(val.shape[0]):
            dMeat = discriminantPrior(val[i], mMean, cov, priorMeat)
            dFat = discriminantPrior(val[i], fMean, cov, priorFat)
            if dMeat > dFat:
                correctMeat += 1
            else:
                wrongFat += 1

        errorRateLDA = (wrongMeat + wrongFat) / (correctMeat + wrongMeat + correctFat + wrongFat)
        errorRatesLDA.append(errorRateLDA)

    return errorRatesLDA

errorRatesLDAPrior = []
for mIm, aIm in allImgs:
    [fPix, fR, fC] = hf.getPix(mIm, aIm[:,:,1])
    [mPix, mR, mC] = hf.getPix(mIm, aIm[:,:,2])
    mMean = np.mean(mPix, 0)
    fMean = np.mean(fPix, 0)
    mCov = np.cov(mPix, rowvar=False)
    fCov = np.cov(fPix, rowvar=False)
    meatM = mPix.shape[0]
    fatM = fPix.shape[0]
    pooledCov = (meatM * mCov + fatM * fCov) / (meatM + fatM)
    errorRatesLDAPrior.append(LDAPrior(mMean, fMean, pooledCov, pMeat, pFat))

for i in range(5):
    for j in range(5):
        print("error rate using day:", days[i], "on day", days[j], "is", errorRatesLDAPrior[i][j])
    print()

# tabular form
for i in range(5):
    for j in range(5):
        if i == j:
            print("X", end="\t")
        else:
            print(round(errorRatesLDAPrior[i][j], 5), end="\t")
    print()

# for comparison, this is the old table without priors
for i in range(5):
    for j in range(5):
        if i == j:
            print("X", end="\t")
        else:
            print(round(errorRatesLDA[i][j], 5), end="\t")
    print()