%%
% 
%  Load the images.
%  
% 
clc; clear;
img1 = imread('img1.jpg');
img2 = imread('img2.jpg');
img3 = imread('img3.jpg');
img4 = imread('img4.jpg');

%% I. Color Representation - 1: RGB
% 
% * A digital image is a Two-Dimensional signal and described by the brightness or color of picture elements ("pixels") indexed by horizontal and vertical coordinates.
% * By default, a color image is stored by MATLAB using 3 matrices, each representing red, green and blue components of pixels. In image/video processing, it is also referred to as R/G/B channels. A matrix is essentially an array indexed by two indexing variables typically for row and column.
% 
%%
% 
% *1.* First image.
% 
img1_r = img1( :, :, 1 ); img1_g = img1( :, :, 2 ); img1_b = img1( :, :, 3 );
figure(5), subplot(2, 2, 1), imshow(img1), title('img1 Original')
subplot(2, 2, 2), imshow(img1_r), title('Red'), colorbar
subplot(2, 2, 3), imshow(img1_g), title('Green'), colorbar
subplot(2, 2, 4), imshow(img1_b), title('Blue'), colorbar

figure(9), subplot(2, 2, 1), imshow(img1), title('Original')
img5 = zeros( size(img1) );  %% initialize arrays of the same size as img1
img5 = uint8( img5 );   %% ensure correct data type of unsigned 8-bit integer
img5(:,:,3) = img1_b;
subplot(2, 2, 2), imshow(img5), title('1 component: Blue')
img5(:,:,1) = img1_r;
subplot(2, 2, 3), imshow(img5), title('2 components: R+B')
img5(:,:,2) = img1_g;
subplot(2, 2, 4), imshow(img5), title('3 components: RGB')
%%
% 
% 
% * This is a picture of a hisstoric Rt.66 doodle taken in Pontiac, IL. From the result we can see that this picture contains multiple colors. Thus, there isn't that big of difference on each of the RGB layer. 
%  
% 

%%
% 
%  PREFORMATTED
% 
% *2.* Second image.
% 
img2_r = img2( :, :, 1 ); img2_g = img2( :, :, 2 ); img2_b = img2( :, :, 3 );
figure(6), subplot(2, 2, 1), imshow(img2), title('img2 Original')
subplot(2, 2, 2), imshow(img2_r), title('Red'), colorbar
subplot(2, 2, 3), imshow(img2_g), title('Green'), colorbar
subplot(2, 2, 4), imshow(img2_b), title('Blue'), colorbar

figure(10), subplot(2, 2, 1), imshow(img2), title('Original')
img6 = zeros( size(img2) );  %% initialize arrays of the same size as img2
img6 = uint8( img6 );   %% ensure correct data type of unsigned 8-bit integer
img6(:,:,3) = img2_b;
subplot(2, 2, 2), imshow(img6), title('1 component: Blue')
img6(:,:,2) = img2_g;
subplot(2, 2, 3), imshow(img6), title('2 components: B+G')
img6(:,:,1) = img2_r;
subplot(2, 2, 4), imshow(img6), title('3 components: RGB')
%%
% 
% * This picture is a sunrise scene taken at Sandy Point State Park, MD. The color of the sunlight in the morning is mainly red, as it shows in the red layer.
% * When combined, the blue and green layer do restore some of the information, but the red layer do most of the jobs on that.
% 
%%
% 
%  PREFORMATTED
% 
% *3.* Third image.
% 
img3_r = img3( :, :, 1 ); img3_g = img3( :, :, 2 ); img3_b = img3( :, :, 3 );
figure(7), subplot(2, 2, 1), imshow(img3), title('img3 Original')
subplot(2, 2, 2), imshow(img3_r), title('Red'), colorbar
subplot(2, 2, 3), imshow(img3_g), title('Green'), colorbar
subplot(2, 2, 4), imshow(img3_b), title('Blue'), colorbar

figure(11), subplot(2, 2, 1), imshow(img3), title('Original')
img7 = zeros( size(img3) );  %% initialize arrays of the same size as img3
img7 = uint8( img7 );   %% ensure correct data type of unsigned 8-bit integer
img7(:,:,2) = img3_g;
subplot(2, 2, 2), imshow(img7), title('1 component: Green')
img7(:,:,1) = img3_r;
subplot(2, 2, 3), imshow(img7), title('2 components: R+G')
img7(:,:,3) = img3_b;
subplot(2, 2, 4), imshow(img7), title('3 components: RGB')
%%
% 
% * This picture is taken at the Tail of the Dragon, NC. The car on the left and the sky are white, thus their values on all three layers are high.
% * The car on the right and the forest are green, thus the green layer itself can restore most of the info of the whole picture.
% 

%%
% 
% *4.* Fourth image.
% 
img4_r = img4( :, :, 1 ); img4_g = img4( :, :, 2 ); img4_b = img4( :, :, 3 );
figure(8), subplot(2, 2, 1), imshow(img4), title('img4 Original')
subplot(2, 2, 2), imshow(img4_r), title('Red'), colorbar
subplot(2, 2, 3), imshow(img4_g), title('Green'), colorbar
subplot(2, 2, 4), imshow(img4_b), title('Blue'), colorbar

figure(12), subplot(2, 2, 1), imshow(img4), title('Original')
img8 = zeros( size(img4) );  %% initialize arrays of the same size as img4
img8 = uint8( img8 );   %% ensure correct data type of unsigned 8-bit integer
img8(:,:,1) = img4_r;
subplot(2, 2, 2), imshow(img8), title('1 component: Red')
img8(:,:,3) = img4_b;
subplot(2, 2, 3), imshow(img8), title('2 components: R+B')
img8(:,:,2) = img4_g;
subplot(2, 2, 4), imshow(img8), title('3 components: RGB')
%%
% 
% * The guy in the red hoodie is one of my friends. For human objects, the color is mainly based on the clothes he/she is wearing. Thus the red layer alone can restore most of the info of the object.
% 


%%  II. Color Representation - 2: HSV

img_hsv1 = rgb2hsv(img1);
img_hsv1_h = img_hsv1( :, :, 1 ); img_hsv1_s = img_hsv1( :, :, 2 ); img_hsv1_v = img_hsv1( :, :, 3 );
figure(13), subplot(2, 2, 1), imshow(img1), title('Original')
subplot(2, 2, 2), imshow(img_hsv1_h), title('Hue'), colorbar
subplot(2, 2, 3), imshow(img_hsv1_s), title('Saturation'), colorbar
subplot(2, 2, 4), imshow(img_hsv1_v), title('Value of Brightness'), colorbar
%%
% 
% From the different HSV layers of the image, it's clearly that:
%
% * As *Hue* varies from 0 to 1, the resulting color varies from red through yellow, green, cyan, blue, and magenta, and returns to red.
%
% * When *Saturation* is 0, the colors are unsaturated (i.e., shades of gray); When it is 1, the colors are fully saturated (i.e., they contain no white component and appear to be most solid/pure.
%
% * As *Value* varies from 0 to 1, the brightness increases. The most of look of the image can be shown in this layer except the color details.
%  
% 

%%
figure(17), subplot(2, 2, 1), imshow(img1), title('Original')
img_hsv5 = zeros( size(img1) );  %% initialize arrays of the same size as img1
img_hsv5 = double( img_hsv5 );   %% ensure correct data type of double
img_hsv5(:,:,1) = img_hsv1_h;
subplot(2, 2, 2), imshow(hsv2rgb(img_hsv5)), title('1 component: Hue')
img_hsv5(:,:,2) = img_hsv1_s;
subplot(2, 2, 3), imshow(hsv2rgb(img_hsv5)), title('2 components: H+S')
img_hsv5(:,:,3) = img_hsv1_v;
subplot(2, 2, 4), imshow(hsv2rgb(img_hsv5)), title('3 components: HSV')
%%
% 
% * In this part, I try to recover the image from the HSV layers using the built-in 'hsv2rgb' function. It shows that only the Hue layer or the Saturation layer, or the combination of these two can't show anything, because the image has no brightness. Only with the brightness layer the picture will not be a dark background.
%  
% 

%%  III. Processing Grayscale Image
%%
% 
% * The Value-of-Brightness component gives a *grayscale* version of the original color image. This value is also known in image processing as the *image intensity* . Some people may call this "black-and-white" pictures, but it actually offers multiple shades of gray instead of just two colors of black and white.
% 
%%
% 
% *1.* First image.
% 
% *1) Setting the thresold* for contrast adjustment. Converting a RGB image into a *grayscale* version. *Flipping the shades* : Change dark to bright, and vice versa; this gives the effect of a negative film
% 

t1 = 20;  t2 = 250;  % set a low and high threshold for contrast adjustment
img9 = rgb2gray(img1);   % convert a color image to grayscale
img1_1 = 255 - img9;
%%
% 
% *2) Clipping and contrast stretching* : Change all pixels below a low grayshade (t1) to black, and above a high shade (t2) to white, and stretch all shades in between. This technique is especially useful for images with low contrast (where pixels that should be dark is not dark enough and/or those that should be bright is not bright enough).
% 
img1_2 = double(img9);   % initialize an array with equal size as img
img1_2 = double((img9(:,:) >= t1) & (img9(:,:) <= t2)) .* ((img1_2(:,:) - t1) * 255 / (t2 - t1) ) + ...
         (img9(:,:) > t2) * 255 + (img9(:,:) < t1) * 0;
img1_2 = uint8(img1_2);  % convert data type back to uint8
%%
% 
% *Bonus Question* : What if t1 = t2? Would it cause "dividing by zero" by the above implementation? What effect would the processing now become? How would you revise the MATLAB script to handle this special case?
%
% *Answer* :
%
% When t1 = t2, it will cause "dividing by zero". However, in the recursion this will cause the numerator part to be zero. Thus value at this index will be NaN. The value will become 0 in the unit8 data type, which will not affect the result.
%
% If t1 = t2, the processing will become set the part lower than t1 to zero and the part higher than t1 to 255. The result will only be an image of pure black and pure white, which is not a good result.
%
% To avoid it, the value of t1 and t2 should be found according to the histogram of the grayscale by finding [t1, t2] where most of the pixels' values locate. For example, most of the pixels' values concentrate at a low gray value. We can set t1 = 50 and t2 = 180 to get a better stretching result.
% 

%%
% 
% *3) Histogram equalization* : enhance contrast by adjusting gray shade distribution to nearly uniform
% 
img1_3 = histeq(img9);  % apply histogram equalization
% Display and compare all results
figure(23), subplot(2, 2, 1), imshow(img9), title('Original Grayscale'), colorbar
subplot(2, 2, 2), imshow(img1_1), title('Negative Film'), colorbar
subplot(2, 2, 3), imshow(img1_2), title('Contrast Clip & Stretch'), colorbar
subplot(2, 2, 4), imshow(img1_3), title('Histogram Equalization'), colorbar

% Display and compare histograms of different processings
figure(24),
subplot(3,1,1), imhist( img9 ), title('Histogram - Original')
subplot(3,1,2), [c2, ~] = imhist( img1_2 );
   stem(0:255, c2, '.'), axis([0,255, 0, max(c2)]),
   title('Histogram - Contrast Clipped'), hold on,
   plot(0, c2(1), 'o', 255, c2(256), 'o'), hold off
subplot(3,1,3), imhist( img1_3 ), title('Histogram - Equalized')
%%
% 
% * The histogram of this grayscale image concentrates at about 20 - 100, and around 250. After contrast clip & stretch, the color features become more obvious. It looks like all the pixels have been shrpened.
% * After histogram equalization, the image looks more natural. The gray shades become more evenly distributed.
% 

%%
% 
% *2.* Second image.
%
t1 = 30;  t2 = 230;  % set a low and high threshold for contrast adjustment
img10 = rgb2gray(img2);   % convert a color image to grayscale
img2_1 = 255 - img10;

img2_2 = double(img10);   % initialize an array with equal size as img9
img2_2 = double((img10(:,:) >= t1) & (img10(:,:) <= t2)) .* ((img2_2(:,:) - t1) * 255 / (t2 - t1) ) + ...
         (img10(:,:) > t2) * 255 + (img10(:,:) < t1) * 0;
img2_2 = uint8(img2_2);  % convert data type back to uint8

img2_3 = histeq(img10);  % apply histogram equalization on img3
% Display and compare all results
figure(27), subplot(2, 2, 1), imshow(img10), title('Original Grayscale'), colorbar
subplot(2, 2, 2), imshow(img2_1), title('Negative Film'), colorbar
subplot(2, 2, 3), imshow(img2_2), title('Contrast Clip & Stretch'), colorbar
subplot(2, 2, 4), imshow(img2_3), title('Histogram Equalization'), colorbar

% Display and compare histograms of different processings
figure(28),
subplot(3,1,1), imhist( img10 ), title('Histogram - Original')
subplot(3,1,2), [c2, ~] = imhist( img2_2 );
   stem(0:255, c2, '.'), axis([0,255, 0, max(c2)]),
   title('Histogram - Contrast Clipped'), hold on,
   plot(0, c2(1), 'o', 255, c2(256), 'o'), hold off
subplot(3,1,3), imhist( img2_3 ), title('Histogram - Equalized')
%%
% 
% * The histogram of this grayscale image concentrates at around 50, and between 140 to 220.
% * The image after contrast clip & stretch and histogram equalization looks more brighter than the original one.
% 

%%
% 
% *3.* Third image.
% 
t1 = 20;  t2 = 250;  % set a low and high threshold for contrast adjustment
img11 = rgb2gray(img3);   % convert a color image to grayscale
img3_1 = 255 - img11;

img3_2 = double(img11);   % initialize an array with equal size as img9
img3_2 = double((img11(:,:) >= t1) & (img11(:,:) <= t2)) .* ((img3_2(:,:) - t1) * 255 / (t2 - t1) ) + ...
         (img11(:,:) > t2) * 255 + (img11(:,:) < t1) * 0;
img3_2 = uint8(img3_2);  % convert data type back to uint8

img3_3 = histeq(img11);  % apply histogram equalization on img3
% Display and compare all results
figure(31), subplot(2, 2, 1), imshow(img11), title('Original Grayscale'), colorbar
subplot(2, 2, 2), imshow(img3_1), title('Negative Film'), colorbar
subplot(2, 2, 3), imshow(img3_2), title('Contrast Clip & Stretch'), colorbar
subplot(2, 2, 4), imshow(img3_3), title('Histogram Equalization'), colorbar

% Display and compare histograms of different processings
figure(32),
subplot(3,1,1), imhist( img11 ), title('Histogram - Original')
subplot(3,1,2), [c2, ~] = imhist( img3_2 );
   stem(0:255, c2, '.'), axis([0,255, 0, max(c2)]),
   title('Histogram - Contrast Clipped'), hold on,
   plot(0, c2(1), 'o', 255, c2(256), 'o'), hold off
subplot(3,1,3), imhist( img3_3 ), title('Histogram - Equalized')
%%
% 
% * The histogram of this grayscale image concentrates at between 20 to 100, and around 240.
% * The image after contrast clip & stretch and histogram equalization looks more brighter than the original one, expecially for the road and the forest.
% 

%%
% 
% * *Bonus task* : Apply some of the above processing to one or multiple RGB color channels of the color image, and assemble back the color image (which consists of three arrays as shown earlier).
% 
img = imread('DIP_lab1_intro_01.png');
img_r = img(:,:,1); img_g = img(:,:,2); img_b = img(:,:,3);
figure(33),
subplot(3,1,1), imhist( img_r ), title('Histogram - R')
subplot(3,1,2), imhist( img_g ), title('Histogram - G')
subplot(3,1,3), imhist( img_b ), title('Histogram - B')
% Flipping the shades
img_r_1 = 255 - img_r; img_g_1 = 255 - img_g; img_b_1 = 255 - img_b;
% Contrast clip and stretch
t1 = 25; t2 = 225;
img_r_2 = double(img_r);  
for i=1:size(img_r, 1)
    for j=1:size(img_r, 2)
        if (img_r(i, j) >= t1) && (img_r(i, j) <= t2)                                        
           img_r_2(i, j) = (img_r_2(i, j) - t1) * 255 / (t2 - t1);
        elseif (img_r(i, j) > t2)
           img_r_2(i, j) = 255;
        else
           img_r_2(i, j) = 0; 
        end
    end
end
img_r_2 = uint8(img_r_2);
t1 = 0; t2 = 250;
img_g_2 = double(img_g);  
for i=1:size(img_g, 1)
    for j=1:size(img_g, 2)
        if (img_g(i, j) >= t1) && (img_g(i, j) <= t2)                                        
           img_g_2(i, j) = (img_g_2(i, j) - t1) * 255 / (t2 - t1);
        elseif (img_g(i, j) > t2)
           img_g_2(i, j) = 255;
        else
           img_g_2(i, j) = 0; 
        end
    end
end
img_g_2 = uint8(img_g_2);
t1 = 0; t2 = 100;
img_b_2 = double(img_b);  
for i=1:size(img_b, 1)
    for j=1:size(img_b, 2)
        if (img_b(i, j) >= t1) && (img_b(i, j) <= t2)                                        
           img_b_2(i, j) = (img_b_2(i, j) - t1) * 255 / (t2 - t1);
        elseif (img_b(i, j) > t2)
           img_b_2(i, j) = 255;
        else
           img_b_2(i, j) = 0; 
        end
    end
end
img_b_2 = uint8(img_b_2);
% Histogram equalization
img_r_3 = histeq(img_r_2); img_g_3 = histeq(img_g_2); img_b_3 = histeq(img_b_2);
% Assemble and Display
img_1 = uint8(zeros(size(img)));    % just red layer
img_2 = uint8(zeros(size(img)));    % just green layer
img_3 = uint8(zeros(size(img)));    % just blue layer
img_4 = uint8(zeros(size(img)));    % r + g
img_5 = uint8(zeros(size(img)));    % r + b
img_6 = uint8(zeros(size(img)));    % g + b
img_7 = uint8(zeros(size(img)));    % r + g + b
img_1(:,:,1) = img_r_3; img_1(:,:,2) = img_g; img_1(:,:,3) = img_b;
img_2(:,:,1) = img_r; img_2(:,:,2) = img_g_3; img_2(:,:,3) = img_b;
img_3(:,:,1) = img_r; img_3(:,:,2) = img_g; img_3(:,:,3) = img_b_3;
img_4(:,:,1) = img_r_3; img_4(:,:,2) = img_g_3; img_4(:,:,3) = img_b;
img_5(:,:,1) = img_r_3; img_5(:,:,2) = img_g; img_5(:,:,3) = img_b_3;
img_6(:,:,1) = img_r; img_6(:,:,2) = img_g_3; img_6(:,:,3) = img_b_3;
img_7(:,:,1) = img_r_3; img_7(:,:,2) = img_g_3; img_7(:,:,3) = img_b_3;

figure(34), subplot(2, 2, 1), imshow(img), title('Original');
subplot(2, 2, 2), imshow(img_1), title('1 Component: R');
subplot(2, 2, 3), imshow(img_2), title('1 Component: G');
subplot(2, 2, 4), imshow(img_3), title('1 Component: B');
figure(35), subplot(2, 2, 1), imshow(img), title('2 Components: R+G');
subplot(2, 2, 2), imshow(img_1), title('2 Components: R+B');
subplot(2, 2, 3), imshow(img_2), title('2 Components: G+B');
subplot(2, 2, 4), imshow(img_3), title('3 Components: RGB');
%%
% 
% * From the result we can see that the change on the blue layer has the biggest impact on the whole image, which is because the original histogram of the blue layer has the smallest region and the stretching helps make the contrast much bigger. For each layer, the change is obvious - the color of the flowers changes with operating only on red layer, and the color of the green leaves changes with operating only on green layer.
% 

%% IV. Green Photography for Fun Background
% In a TV weather report, the viewer sees a weatherman standing in front of a map or some other computer-generated image. In reality, the weatherman is standing in front of a green (or blue) curtain. A special camera system will carry out processing and extract the weatherman?s image, and superimpose it upon a computer-generated weather related image.
% The basic idea of this camera system is to split an image into RGB channels, create a mask based on the information of blue channel, and use this mask to extract the desired image.

img13 = imresize(imread('Photo_Gomez.jpg'), 0.25);
img13_r = img13( :, :, 1 ); img13_g = img13( :, :, 2 ); img13_b = img13( :, :, 3 );
img14 = img13;
[m, n, ~] = size(img13);
mask = zeros(m, n);
for i = 1 : m
    for j = 1 : n
        if ((img13_g(i, j) >= 131) && (img13_g(i, j) <= 162) && (img13_b(i, j) <= 108))  && (img13_r(i, j) <= 89) ...
                || ((img13_g(i, j) > 83) && (img13_g(i, j) <= 134) && (img13_b(i, j) < 108))   && (img13_r(i, j) < 82) ...
                || ((img13_g(i, j) >= 55) && (img13_g(i, j) <= 100) && (img13_b(i, j) <= 50))   && (img13_r(i, j) < 59) ...
                || ((img13_g(i, j) >= 28) && (img13_g(i, j) <= 55) && (img13_b(i, j) <= 30))   && (img13_r(i, j) < 27) ...
                || ((img13_g(i, j) >= 97) && (img13_g(i, j) <= 134) && (img13_b(i, j) <= 88))   && (img13_r(i, j) <= 112)
            img14(i, j, :) = 0;
            mask(i, j) = 255;
        else
            img14(i, j, :) = img13(i, j, :);
        end
    end
end

img15 = imresize(imread('EE101_lab1_images\fall_color.jpg'), 1);
[m1, n1, ~] = size(img15);
m2 = m1 - m; n2 = n1 - n - 50; 
for i = 1 : m
    for j = 1 : n
        if mask(i, j) == 0
            img15(i + m2, j + n2, :) = img13(i, j, :);
        end
    end
end
figure(36)
subplot(1,3,1), imshow(img13), title('Original Photo')
subplot(1,3,2), imshow(mask), title('Identify Green Background')
subplot(1,3,3), imshow(img15), title('with New Background')

%%
% 
%  The method I used here is simply observing and choose the different thresholds for different occasions. There're still some imperfections (several pixels) on the border of the mask and the human object, but only visible when the image is largened enough.
% 
% 









