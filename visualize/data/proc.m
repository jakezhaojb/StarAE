% This script is used to prepare images.

clc
clear

IMAGES = zeros(512, 512, 25);

for i = 1: 10
    a = imread(strcat(int2str(i), '.png'));
    a = double(a) / 255;
    IMAGES(:,:,i) = a;
end

for i = 11: 25
    disp(i)
    a = imread(strcat(int2str(i), '.png'));
    if size(a, 3) > 1
        a = rgb2gray(a);
    end
    a = imresize(a, [512, 512]);
    a = double(a) / 255;
    IMAGES(:,:,i) = a;
end

save('IMAGES_25.mat', 'IMAGES')
