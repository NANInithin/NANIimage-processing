
[filename,pathname] = uigetfile('.','select input image');
filewithpath = strcat(pathname,filename);
imgt = imread(filewithpath);
CS = [16,16];
[hogfvt,hogvist] = extractHOGFeatures(imgt,'cellsize',CS);
predictedLabel = predict(classifier,hogfvt);
figure
imshow(imgt)
title(['shape recognized is' char(predictedLabel)]);