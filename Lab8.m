%% Hybrid Computer Vision Pipeline: Traditional + AI
% -------------------------------------------------
% 1. Load image & pre-trained CNN (SqueezeNet)
% 2. Traditional: K-means color segmentation → isolate object
% 3. AI: Classify both original and isolated image
% 4. Compare results: confidence & accuracy
% -------------------------------------------------

close all; clear; clc;

%% 1. Load image and AI network
img = imread('peppers.png');           % 384x512x3
net = squeezenet;                      % 1000-class CNN, input: 227x227x3
inputSize = net.Layers(1).InputSize(1:2);

fprintf('Loaded SqueezeNet (input size: %dx%d)\n', inputSize);

%% 2. Traditional Pre-processing: K-means Segmentation
fprintf('Running traditional K-means segmentation...\n');
imgLab = rgb2lab(img);
ab = im2single(imgLab(:,:,2:3));
nClusters = 2;
pixelLabels = imsegkmeans(ab, nClusters);

% Choose the cluster with higher average intensity (usually the object)
meanIntensity = zeros(1, nClusters);
for k = 1:nClusters
    meanIntensity(k) = mean(imgLab(pixelLabels == k));
end
objectCluster = find(meanIntensity == max(meanIntensity), 1);
mask = pixelLabels == objectCluster;

% Create isolated object
isolated = img;
isolated(repmat(~mask, [1 1 3])) = 0;

%% 3. Resize both images for CNN
imgResized = imresize(img, inputSize);
isolatedResized = imresize(isolated, inputSize);

%% 4. AI Classification: Original vs. Isolated
[labelOrig, scoresOrig] = classify(net, imgResized);
[labelIso, scoresIso] = classify(net, isolatedResized);

confOrig = max(scoresOrig);
confIso = max(scoresIso);

%% 5. Display Results
figure('Name', 'Hybrid Pipeline: Traditional + AI', 'Position', [100, 100, 1000, 600]);

subplot(2,3,1); imshow(img);           title('Original Image');
subplot(2,3,2); imshow(mask);          title('K-means Mask');
subplot(2,3,3); imshow(isolated);      title('Isolated Object');

subplot(2,3,4); imshow(imgResized);    title(sprintf('AI Input (Original)\n"%s"', string(labelOrig)));
subplot(2,3,5); imshow(isolatedResized); title(sprintf('AI Input (Isolated)\n"%s"', string(labelIso)));

% Confidence bar chart
subplot(2,3,6);
bar([confOrig, confIso]);
set(gca, 'XTickLabel', {'Original', 'Isolated'}, 'YLim', [0 1]);
ylabel('Confidence');
title('Classification Confidence');
grid on;

%% 6. Print Results
fprintf('\n=== HYBRID PIPELINE RESULTS ===\n');
fprintf('Original Image  → Class: %-15s | Conf: %.2f%%\n', string(labelOrig), confOrig*100);
fprintf('Isolated Object → Class: %-15s | Conf: %.2f%%\n', string(labelIso),  confIso*100);

if confIso > confOrig
    fprintf('→ Traditional pre-processing IMPROVED confidence by %.2f%%\n', (confIso - confOrig)*100);
else
    fprintf('→ No improvement from segmentation.\n');
end

% Save figure for README
exportgraphics(gcf, 'results/hybrid_pipeline.png', 'Resolution', 150);
fprintf('\nResult image saved: results/hybrid_pipeline.png\n');