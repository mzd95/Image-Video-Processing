%% LAB 2: Feature Detection
% 
% Updated: v1. 2019-2-25.
% By Zhongdao Mo (mzd95@terpmail.umd.edu)
% 

%% I. Visual Quantization and Dithering [1]
% 
%  The number of bits used to represent the image can be reduced by implementing visual quantization. However, some additional contours could appear on the image.
% 
%%
% 
%  Function rgb2ind() is used to convert a RGB image to an indexed image uisng minmum variance quantization. If we set the parameter to 'nodither', this function will only do quantization and there will be visible contours on the image. If we set the 
%  parameter to 'dither', then this function do both quantization and dithering and the result will be smoother.
% 

img1 = imresize(imread('Photo_Gomez.jpg'), 0.25);
[img1_no_dithering,map1] = rgb2ind(img1,16,'nodither');
[img1_dithering,map2] = rgb2ind(img1,16,'dither');
figure;
subplot(1,2,1);imshow(img1_no_dithering,map1);
subplot(1,2,2);imshow(img1_dithering,map2);
%%
% 
%  In the non-dithered image, there's visible contour where the color graudally changes. In the dithered image, we can see that there're mixed pixels with different gray-scale values instead of a sudden change of the gray-scale value.
% 

img2 = rgb2gray(imread('lena.jpg'));
img2_1 = uint8(zeros(size(img2,1),size(img2,2),3));
img2_1(:,:,1) = img2(:,:); img2_1(:,:,2) = img2(:,:); img2_1(:,:,3) = img2(:,:);
[img2_no_dithering,map1] = rgb2ind(img2_1,4,'nodither');
[img2_dithering,map2] = rgb2ind(img2_1,4,'dither');
figure(2);
subplot(1,2,1);imshow(img2_no_dithering,map1);
subplot(1,2,2);imshow(img2_dithering,map2);
%%
% 
%  Similar results can be seen in this example.
% 


%% II. Morphological Filters
% 
%  Morphological operations are non-linear operations related to the shape or morphology of features in an image. They often take a binary image and a structuring element as input and combine them using a set operator (such as intersection, union, 
%  inclusion, complement). These operations process objects in the input image based on characteristics of its shape, which are encoded in the structuring element. [2]
% 
%%
% 
% *1) Dilation & Erosion*
% 
%  The effect of dilation is to gradually enlarge the boundary of reginos of foreground pixels.
%
%  The effect of erosion is to erode away the boudary of the regions of foreground pixels. The area of foreground pixels will thus shrink.
% 

% Set the foreground image
FG = zeros(14,14); FG(4:10, 4:10) = 1;
% Set different types of structuring elements.
SE1 = strel('line',4,30); % 30 degrees counterclockwise from horizontal
SE2 = strel('square',3);
SE3 = strel('octagon',3);
SE4 = strel('rectangle',[2,3]);
SE5 = strel('diamond',2);
% dilation & erosion
FG1 = imdilate(FG,SE1);
FG2 = imdilate(FG,SE2);
FG3 = imdilate(FG,SE3);
FG4 = imdilate(FG,SE4);
FG5 = imdilate(FG,SE5);
FG6 = imerode(FG,SE3);
FG7 = imerode(FG,SE4);
FG8 = imerode(FG,SE5);
figure(3);
subplot(3,3,1),imshow(FG),title('Original');
subplot(3,3,2),imshow(FG1),title('line SE - dilate');
subplot(3,3,3),imshow(FG2),title('square SE - dilate');
subplot(3,3,4),imshow(FG3),title('octagonal SE - dilate');
subplot(3,3,5),imshow(FG4),title('rectangle SE - dilate');
subplot(3,3,6),imshow(FG5),title('diamond SE - dilate');
subplot(3,3,7),imshow(FG6),title('octagonal SE - erode');
subplot(3,3,8),imshow(FG7),title('rectangle SE - erode');
subplot(3,3,9),imshow(FG8),title('diamond SE - erode');
%%
% 
% *2) Boundary Extraction*
% 
% 
%  The boundary of a set A, can be obtained by first eroding A by B (where B is a suitable structuring element) and then performing the set differennce between a and its erosion. [3]
% 

% Here we use the mask of professor Gomez obtained in web journal 1
img4 = imread('mask_Gomez.png');
img4_1(:,:,1) = img4(:,:);img4_1(:,:,2) = img4(:,:);img4_1(:,:,3) = img4(:,:);
img4_g = rgb2gray(img4_1);
se = strel('disk',4);
% Single pixel boudary using the built-in MATLAB function.
img4_bd = bwperim(img4_g);
% Thicker perimeter by erosion.
img4_bd2 = img4_g - imerode(img4_g,se);
figure(4)
subplot(1,3,1),imshow(img4_g),title('Original');
subplot(1,3,2),imshow(img4_bd),title('Single-pixel');
subplot(1,3,3),imshow(img4_bd2),title('Eroded boudary');
%%
% 
%  The result shows a similar but thicker perimeter sompared to the built-in MATLAB function.
% 


%%
% 
% *3) Morphological Gradient* [4]
% 
% 
%  Another way to emphasize the contours in a gray-scale image is to calculate the mophological gradient. Flat structuring elements have all zero height values and are thus specified entirely by the neighbourhood.Local minimum and maximum can be 
%  calculated by dilation and erosion with flat structuring elements, and the difference between the to is the morphological gradient which emphesizes the contours. [3]
% 

img5 = imread('cameraman.tif');
% Flat 2x2 structuring element
se = strel(ones(2));
img5_max = imdilate(img5,se);
img5_min = imerode(img5,se);
img5_grad = img5_max - img5_min;
figure(5)
subplot(2,2,1),imshow(img5),title('Original');
subplot(2,2,2),imshow(img5_max),title('Dilation');
subplot(2,2,3),imshow(img5_min),title('Erosion');
subplot(2,2,4),imshow(img5_grad),title('Gradient');
%%
% 
%  The result shows that the method does extract the contours where the color level changes. The quicker the color level changes, the thicker the contour will be.
% 


%% III. Image Smoothening & Sharpening
% 
%  The process of image smoothening & sharpening is basically convolve the original image with different types of filters.
%
% *1) Averaging filters*, which are low-pass filters are used for image smoothening. They set each pixel of the output image as the weighted average of the original pixel and its' neibourboods'. Such operation passes the low frequency componentsof the original image. [5]
% 
img3 = rgb2gray(imread('forest.jpg'));
avg_filter1 = ones(3,3)/9;
avg_filter2 = ones(3,3)/16; avg_filter2(2,2) = 0.5;

img3_LP1 = uint8(filter2(avg_filter1, img3));
img3_LP2 = uint8(filter2(avg_filter2, img3));
figure(6);
subplot(1,3,1), imshow(img3), title('Original');
subplot(1,3,2), imshow(img3_LP1), title('Averaging filter 1');
subplot(1,3,3), imshow(img3_LP2), title('Averaging filter 2');
%%
% 
%  The first filter smooths and blurs the image more than the second filter, because the second one has more weight on the center pixel.
% 
% *2) High-pass filters* are used for image sharpening.
% 
img3_HP = double(img3) - double(img3_LP1);
img3_1 = double(img3) + double(img3_HP) * 1;
img3_SHP = uint8((img3_1 > 255) * 255 + (img3_1 >= 0 & img3_1 <= 255) .* img3_1);
figure(7);
subplot(1,3,1), imshow(img3), title('Original');
subplot(1,3,2), imshow(img3_HP), title('Highpass Filter');
subplot(1,3,3), imshow(img3_SHP), title('Sharpened');
%%
% 
%  The sharpened image does have more visible contours, while emphasizing the leaves on the ground.
% 


%% IV. Image Enhancement
% 
% *1) Types of noise* [6]
% 
% * Salt & pepper noise: isolated extreme-valued(white/black) pixels spread randomly over the image.
% * Gaussian noise: each pixel in the image will be changed from its original value by a (usually) small amount. A histogram, a plot of the amount of distortion of a pixel value against the grequency with which it occurs, show a normal distribution of noise.
% 
%  Noise at different pixels can be either correlated or uncorrelated; in many cases, noise values at different pixels are modeled as being independent and identically distributed, and hence uncorrelated.
% 
% *2) Noise reduction methods*
% 
% * Linear smoothing filter: it represents a low-pass filter or smoothing operation by setting each pixel to the average value, or a weighted average, of itself and its nearby neighbors. It tends to blur an image.
% * Median filter: it's a non-linear filter. For each pixel, it first sort the neighbouring pixels upon the intensities, and then replace the original value of the pixel with the median value from the list. It's good at removing salt and pepper noise, and also cause relatively little blurring of edges.
% * Wavelet transform: In the wavelet domain, the noise is uniformly spread throughout coefficients while most of the image information is concentrated in a few large ones. Thus, methods based on thresholding of detail subbands coefficients can be used to reduce the noise.
% * Block-matching: it groups similar image fragments into overlapping marcoblocks of identical size, and filter them together in the transform domain. Weighted average of the overlapping pixels is used to restore the image fragments.
% 
%  There're several other methods such as non-local means, statistical method, randomfield and even deep learning.
% 
% *3)* Salt & pepper noise removal
% 
%  Here I try to add salt & pepper noise on the original image and then remove it, both using built-in MATLAB functions.
% 
img6 = rgb2gray(imread('coins.jpg'));
% add salt & pepper noise
img6_noisy = imnoise(img6,'salt & pepper',0.02);
% use a 2x2 median filter to filter out the noise
img6_med = medfilt2(img6_noisy, [3,3]);
figure(8);
subplot(1,3,1), imshow(img6), title('Original');
subplot(1,3,2), imshow(img6_noisy), title('S&P noise');
subplot(1,3,3), imshow(img6_med), title('Median-filtered');
%%
% 
%  In the result, the noise is removed successfully. However the median filter also blurs the image a little, which can be seen on the coins that the letters are less visible than those on the original image.
% 
% *4)* Old school image enhancement [7]
% 
%  This old image has some blur on it and salt & pepper type of noise. So here I use sharpening and a median filter to enhance the image.
% 
img10 = rgb2gray(imread('school.jpg'));
hp_filter1 = zeros(3,3); hp_filter1(2,2) = 2;
hp_filter1(1,2) = -1/4; hp_filter1(2,1) = -1/4; hp_filter1(2,3) = -1/4; hp_filter1(3,2) = -1/4;
img10_1 = uint8(filter2(hp_filter1,img10));
img10_2 = medfilt2(img10_1,[2,2]);
figure(9);
subplot(2,2,1),imshow(img10),title('Original');
subplot(2,2,2),imshow(img10_1),title('Sharpened');
subplot(2,2,3),imshow(img10_2),title('Median-filtered');

%%
% 
% *5)* Bright image enhancement
% 
%  This image's histogram is very concentrated at the high values. So here I use histogram equalization and then a high-pass filter to enhance it.
% 
img7 = imread('631Lab1-brightim.jpg');
img7_eq = histeq(img7);
hp_filter2 = ones(3,3); hp_filter2(2,2) = 2;
hp_filter2(1,2) = -1/8; hp_filter2(2,1) = -1/8; hp_filter2(2,3) = -1/8; hp_filter2(3,2) = -1/8;
hp_filter2(1,1) = -1/8; hp_filter2(1,3) = -1/8; hp_filter2(3,1) = -1/8; hp_filter2(3,3) = -1/8;
img7_HP = uint8(filter2(hp_filter2,img7_eq));
figure(10);
subplot(1,3,1),imshow(img7),title('Original');
subplot(1,3,2),imshow(img7_eq),title('Histogram equalized');
subplot(1,3,3),imshow(img7_HP),title('Sharpened');

%%
% 
%  After histogram equalization, the image becomes darker and more features become more visible. However the lower part is still somehow blurred. The high-pass filter does enhance the features a little bit but not perfect.
% 
% *6)* Dark image enhancement
% 
%  This image's histogram concentrates at the low values. However, if I use the built-in function to do histogram equalization on it, the image will become too bright and there'll be much noise. So I use contrast-clip here and set the upper threshold 
%  lower than the maximum value, so that the image doesn't become too bright. A high-pass filter is used to remove the noise.
% 
img8 = imread('631Lab1-darkim.jpg');
t1 = 15;  t2 = 52;  % set a low and high threshold for contrast adjustment
img8_1 = double(img8);   % initialize an array with equal size as img9
img8_1 = double((img8(:,:) >= t1) & (img8(:,:) <= t2)) .* ((img8_1(:,:) - t1) * 255 / (t2 - t1) ) + ...
         (img8(:,:) > t2) * 255 + (img8(:,:) < t1) * 0;
img8_1 = uint8(img8_1);  % convert data type back to uint8
img8_LP = filter2(avg_filter1,img8_1);
img8_HP = double(img8_1) - double(img8_LP);
img8_2 = double(img8_1) + double(img8_HP);
img8_SHP = uint8((img8_2 > 255) * 255 + (img8_2 >= 0 & img8_2 <= 255) .* img8_2);
img8_eq = histeq(img8_1);
figure(11),
subplot(2,2,1), imshow(img8), title('Original')
subplot(2,2,2), imshow(img8_1), title('Contrast Clipped')
subplot(2,2,3),imshow(img8_SHP),title('Sharpened');
subplot(2,2,4), imshow(img8_eq), title('Histogram equalized')
%%
% 
%  I believe the result of contrast-clipped then sharpened image is better than the histogram equalized one, because many of the objects in the latter one are still not visible because they're too bright.
% 
% *7)* Lena enhancement
% 
%  The main noise on this image is salt & pepper noise, so I use a median filter to remove the noise. Because the filter also blurs the image a little, a high-pass filter is used for deblurring.
% 
img9 = imread('631Lab1-noisyLena.tif');
figure(12);
subplot(1,3,1), imshow(img9), title('Original');
img9_1 = medfilt2(img9, [4,4]);
subplot(1,3,2), imshow(img9_1), title('Median-filtered');
% Built-in MATLAB function to sharpen the image
img9_3 = imsharpen(img9_1);
img9_SHP = uint8((img9_2 > 255) * 255 + (img9_2 >= 0 & img9_2 <= 255) .* img9_2);
subplot(1,3,3), imshow(img9_3), title('Sharpened');
%%
% 
%  The median-filtered image has much less noise than the original one. And the high-pass filter helps enhance the features on the image, for example the texture of the hair.
% 

%% References
% 
% *[1]* Image by prof Gomez.
% 
% *[2]* https://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm
% 
% *[3]* Digital image processing. R. Gonzalez, and R. Woods. Prentice Hall, Upper Saddle River, N.J., (2008)
% 
% *[4]* Image by: https://www.researchgate.net/figure/Cameramantif_fig2_241630365
% 
% *[5]* Image by: http://www.digital-photo-secrets.com/tip/2857/how-contrast-affects-your-photos/
% 
% *[6]* https://en.wikipedia.org/wiki/Noise_reduction
% 
% *[7]* Image by: https://www.chicagotribune.com/news/local/breaking/ct-met-old-town-school-petition-20181026-story.html
% 

