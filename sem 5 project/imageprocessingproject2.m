path1 =("C:\Users\NANI NITHIN\Downloads\imagedb\trainimg") 
path2=("C:\Users\NANI NITHIN\Downloads\imagedb\testimg")
traindb = imageDatastore(path1,'IncludeSubfolders',true,'LabelSource','foldernames');
testdb =  imageDatastore(path2,'IncludeSubfolders',true,'LabelSource','foldernames');

img = readimage(traindb,1);
CS = [16,16];
[hogfv,hogvis] = extractHOGFeatures(img,'CellSize', CS);
hogfeaturesize = length(hogfv);
totaltrainimages = numel(traindb.Files);
trainingfeatures = zeros(totaltrainimages, hogfeaturesize, 'single');
for i = 1:totaltrainimages
img = readimage(traindb,i);
trainingfeatures(i, :) = extractHOGFeatures(img,'CellSize',CS);
end
traininglabels = traindb.Labels;
classifier = fitcecoc(trainingfeatures, traininglabels);
totaltestimages = numel(testdb.Files);
testfeatures = zeros(totaltestimages, hogfeaturesize, 'single');
for i = 1:totaltestimages
imgt = readimage(testdb,i);
testfeatures(i, :) = extractHOGFeatures(imgt,'CellSize',CS);
end
testlabels = testdb.Labels;
predictedlabels = predict(classifier, testfeatures);
accuracy = (sum(predictedlabels == testlabels)/ numel(testlabels))*100
plotconfusion(testlabels, predictedlabels)
