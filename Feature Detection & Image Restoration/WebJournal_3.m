%% LAB 3: Feature Detection & Image Restoration
% 
% Updated: v1. 2019-3-12
% By Zhongdao Mo (mzd95@terpmail.umd.edu)
% 

%% I. Canny Edge Detection (REQUIRED TASK)
% 
%  Canny edge detection is a technique to extract useful structural information from different vision objects and dramatically reduce the amount of data to be processed.
% 
% 
%  The basic objectrives for Canny edge detection include: [1]
% 
% * _Low error rate_ . The edges detected must be as close as possible to the
% true edges.
% * _Edge points should be well localized_ . The distance between a point
% marked as an edge by the detector and the center of the true edge should
% be minimum.
% * _Single edge point response_. The number of local maxima around the
% true edge should be minimumm which means that the detector should not
% identify multiple edge pixels where only a single edge point exists.
% 


%%
% 
%  The diagram of the Canny edge detection process is shown below.
% 

canny = imread('canny.jpg');
figure(1), imshow(canny), title('Canny edge detection');

%%
% 
% *1) Gaussian smoothing*
% 
% 
%  First, a Gaussian filter is applied to the image to remove noise. Here I use a 5x5 gaussian filter with sigma of 1.5.
% 

lena0 = rgb2gray(imread('lena.jpg'));
lena = double(lena0);
% size of gaussian filter is 5x5, sigma = 1.5
gaussian = fspecial('gaussian', [5, 5], 1.5);
lena_g = filter2(gaussian, lena);

%%
% 
% *2) Calculate intensity gradients*
% 
% 
%  Here I use Prewitt filter to help get the intensity gradient of the image in both magnitude and direction for each pixel.
% 

% use prewitt filter to obtain the gradients
h_x = fspecial('prewitt'); h_y = h_x.';
lena_x = filter2(h_x, lena_g);
lena_y = filter2(h_y, lena_g);
lena_mag = sqrt(lena_x.^2 + lena_y.^2);
lena_dir = atan2(lena_x, lena_y) * 180 / pi;

%%
% 
% *3) Nonmaximum supression*
% 
% 
%  Set four basic edge directions for a 3x3 region. For each pixel, do the following nonmaximum supression:
%  Find the direction that's closest to the pixel; 
%  If the maginitude of this pixel is less than at least one of its two neighbors on that direction, set the magimitude to zero.
% 

[m, n] = size(lena_g);
lena_nms = zeros(m, n);
for i = 1 : m
    for j = 1 : n
        % horizontal edge
        if((lena_dir(i, j) < -157.5) || (lena_dir(i, j) >= 157.5) || ((lena_dir(i, j) >= -22.5) && (lena_dir(i, j) < 22.5)))
           lena_dir(i, j) = 0;
           if(i ~= 1 && i ~= m && j ~= 1 && j ~= n)
              lena_nms(i, j) = lena_mag(i, j) == max([lena_mag(i, j - 1), lena_mag(i, j), lena_mag(i, j + 1)]);
           end
        end
        % -45 degree edge
        if(((lena_dir(i, j) >= -157.5) && (lena_dir(i, j) < 112.5)) || ((lena_dir(i, j) >= 22.5) && (lena_dir(i, j) < 67.5)))
           lena_dir(i, j) = 45;
           if(i ~= 1 && i ~= m && j ~= 1 && j ~= n)
              lena_nms(i, j) = lena_mag(i, j) == max([lena_mag(i + 1, j - 1), lena_mag(i, j), lena_mag(i - 1, j + 1)]);
           end
        end
        % vertical edge
        if(((lena_dir(i, j) >= -112.5) && (lena_dir(i, j) < -67.5)) || ((lena_dir(i, j) >= 67.5) && (lena_dir(i, j) < 112.5)))
           lena_dir(i, j) = 90;
           if(i ~= 1 && i ~= m && j ~= 1 && j ~= n)
              lena_nms(i, j) = lena_mag(i, j) == max([lena_mag(i - 1, j), lena_mag(i, j), lena_mag(i + 1, j)]);
           end
        end
        % +45 dgree edge
        if(((lena_dir(i, j) >= -67.5) && (lena_dir(i, j) <= -22.5)) || ((lena_dir(i, j) >= 112.5) && (lena_dir(i, j) < 157.5)))
           lena_dir(i, j) = 135;
           if(i ~= 1 && i ~= m && j ~= 1 && j ~= n)
              lena_nms(i, j) = lena_mag(i, j) == max([lena_mag(i - 1, j - 1), lena_mag(i, j), lena_mag(i + 1, j + 1)]);
           end
        end
    end
end
lena_nms = lena_nms .* lena_mag;

%%
% 
% *4) Thresholding & connectivity*
% 
% 
%  Use Hysteresis thresholding which uses two thresholds to eliminate false edge points. Then 8-connectivity is applied to fill the gap between strong edge points.
% 

t_low = 0.05; t_high = 0.1;
t_low = t_low * max(max(lena_nms));
t_high = t_high * max(max(lena_nms));
lena_edge = zeros(m, n);
for i = 1 : m
    for j = 1 : n
        if(lena_nms(i, j) < t_low)
            lena_edge(i, j) = 0;
        elseif(lena_nms(i, j) > t_high)
            lena_edge(i, j) = 1;
        % 8-connectivity 
        elseif(lena_nms(i - 1, j - 1) > t_high || lena_nms(i - 1, j) > t_high || lena_nms(i - 1, j + 1) > t_high || lena_nms(i, j - 1) > t_high || ...
                lena_nms(i, j + 1) > t_high || lena_nms(i + 1, j - 1) > t_high || lena_nms(i + 1, j) > t_high || lena_nms(i + 1, j + 1) > t_high)
            lena_edge(i, j) = 1;
        end
    end
end
lena_edge = uint8(lena_edge .* 255);

%%
% 
% *5) Compare with the built-in function*
% 

lena_edge1 = edge(lena, 'canny', [0.05, 0.1], 1.5);
figure(2);
subplot(1,3,1),imshow(lena0),title('Original');
subplot(1,3,2),imshow(lena_edge),title('Step-by-step Canny');
subplot(1,3,3),imshow(lena_edge1),title('Built-in Canny');

%% II. Detecting Circles [2]
%%
% 
% *1) Determine raduis range of the circles*
% 
% 
%  Here we use the built-in fuction imfindcircles which uses phase coding method to find the circles. However it needs a radius range to do this. The imdistline tool provides a way for us to get a more precise estimate.
% 

chips = imread('coloredchips.png');
%d = imdistline;
%delete(d);

%%
% 
% *2) Initial attempt*
% 
% 
%  The background is quite bright in this image and most of the chips are darker than the background.
%  By default, imfindcircles finds circular objects that are brighter than the background. Set the parameter 'ObjectPolarity' to 'dark' in imfindcircles to search for dark circles.
% 

chips_gray = rgb2gray(chips);
[centers,radii] = imfindcircles(chips, [20 25],'ObjectPolarity','dark')

%%
% 
% *3) Increase detection sensitivity*
% 
% 
%  The output of the last step shows that no circles were find. So the sensitivity level should be raised to detect these circles.
% 

[centers,radii] = imfindcircles(chips, [20 25],'ObjectPolarity','dark', 'Sensitivity', 0.95);
figure(3), imshow(chips)
h = viscircles(centers,radii);

%%
% 
% *4) Use two-stage for finding circles*
% 
% 
%  There's another method called the two-stage method in this function to find the circles. 
% 

[centers,radii] = imfindcircles(chips, [20 25],'ObjectPolarity','dark', 'Sensitivity',0.90,'Method','twostage');
delete(h)
h = viscircles(centers,radii);
%%
% 
%  Note that the two-stage method can detect as many circles with lower sensitivity level than the phase coding method. However, the latter one is faster and more robust to noise than the first method.
% 

%%
% 
% *5) Find bright circles in the image*
% 
% 
%  There're still several yellow chips which can't be found because of the low contrast to the background. Changing the 'ObjectPolarity' to 'bright' can help detect those circles with low contrast.
% 

[centersBright,radiiBright] = imfindcircles(chips, [20 25], 'ObjectPolarity','bright','Sensitivity',0.92);
figure(4), imshow(chips)
hBright = viscircles(centersBright, radiiBright,'Color','b');

%%
% 
% *6) Lower the value of 'EdgeThreshold' & draw all the circles*
% 
% 
%  The boundary pixels of those yellow chips are expected to have low gradient values, which is the reason the function cannot detect them. Lowering the 'EdgeThreshold' will ensure most of the yellow chips can be detected.
% 

[centersBright,radiiBright,metricBright] = imfindcircles(chips, [20 25], 'ObjectPolarity','bright','Sensitivity',0.92,'EdgeThreshold',0.1);
delete(hBright)
hBright = viscircles(centersBright, radiiBright,'Color','b');
h = viscircles(centers,radii);

%% III. Image Restoration - w/ Original Image
%%
% 
% * Image restoration is the operation of taking a noisy image and
% estimating the clean, original image. When the transfer function H is
% known, two types of deconvolution method aree commonly used - inverse
% filtering and Wiener deconvolution. [3]
% 


%%
% 
% *1) Inverse filtering & pseudo-inverse filtering*
% 
% 
%  The typical inverse filter G of H is one such that the sequence of applying H then G to an image results in the original one. In the 2-D frequency domain, the relation between H and G is G(w1, w2) = 1 / H(w1, w2).
% 

flower = imread('flower.tif');
flower_blur = imread('flower1.tif');
% transform to frequency domain
flower_fft = fft2(im2double(flower));
flower_blur_fft = fft2(im2double(flower_blur));
% calculate distortion function H
h = flower_blur_fft ./ flower_fft;
% apply inverse filtering to the blurred image
g = 1 ./ h;
flower_inverse = real(ifft2(g .* flower_blur_fft));
flower_inverse_uint8 = uint8(255 .* flower_inverse);
% calculate MSE
err_filtered = immse(flower,flower_blur)
err_inverse = immse(flower,flower_inverse_uint8)

%%
% 
%  However, there'll be cases when H takes very small values which causes the information on those frequecy loses. If we take very large value of G on those points, the noise will also be amplified which causes more distortion.
%  In order to avoid this, the pseudo-inverse filter can be used, which set G to be zero when the value of H on this frequency is smaller than a threshold value.
% 

% apply pseudo-inverse filtering to the blurred image
flower_pseudo_inverse = real(ifft2((abs(h) > 0.2) .* flower_blur_fft .* g));
flower_pseudo_uint8 = uint8(255 .* flower_pseudo_inverse);
figure(5); subplot(2,2,1), imshow(flower), title('Original image');
subplot(2,2,2), imshow(flower_blur), title('Blurred image');
subplot(2,2,3), imshow(flower_inverse_uint8), title('Inverse filtering');
subplot(2,2,4), imshow(flower_pseudo_uint8), title('Pseudo-inverse filtering');
% calculate MSE
err_pseudo = immse(flower,flower_pseudo_uint8)

%%
% 
%  Note that the result of inverse filtering actually gives the perfect image, while the pseudo-inverse filtering still gives a blurred one.
%  This is because we didn't add noise to the image, and there's only a transfer function in the system. The pseudo-inverse filtering causes loss of information on those zero-value points.
%  To better show the result of the pseudo-inverse filter, zero-mean gaussian noise is added to the blurred image.
% 

flower2 = imnoise(flower_blur, 'gaussian', 0, 0.01);
flower2_fft = fft2(im2double(flower2));
% if we know the transfer function h
flower2_inverse = real(ifft2(g .* flower2_fft));
flower2_inverse_uint8 = uint8(255 .* flower2_inverse);
flower2_pseudo_inverse = real(ifft2((abs(h) > 0.4) .* flower2_fft .* g));
flower2_pseudo_uint8 = uint8(255 .* flower2_pseudo_inverse);
% plot the figure
figure(6); subplot(2,2,1), imshow(flower), title('Original image');
subplot(2,2,2), imshow(flower2), title('Noisy and blurred image');
subplot(2,2,3), imshow(flower2_inverse_uint8), title('Inverse filtering');
subplot(2,2,4), imshow(flower2_pseudo_uint8), title('Pseudo-inverse filtering');
% calculate MSE
err_noisy = immse(flower,flower2)
err_inverse2 = immse(flower,flower2_inverse_uint8)
err_pseudo2 = immse(flower,flower2_pseudo_uint8)
%%
% 
%  It's obvious that the inverse filter couldn't restore the image and results in a bigger MSE because of the noise amplification.
%  The pseudo-inverse filter gives a much better result with less than half of the MSE of the blurred & noisy image.
% 


%%
% 
% *2) Wiener filtering*
% 
% 
%  As we can see in the last part, the inverse filter handles the case when there's no noise very well but does a very poor job with noise; while the pseudo-inverse filter does very well with noise but not without noise.
% 
% 
% * To make a trade-off between deconvolution and noise reduction, Wiener filter is introduced. The idea is to minimize the MSE between the original signal I[n1, n2] and restored signal I'[n1, n2]. This can be transferred to an estimation problem, which estimates the expectation of I[n1, n2] with the observed signal. [4]
%  The filter is not linear. The filter can be expressed as G(w1, w2) = 1 / (H + Snn / (H*Suu)), where Snn is the PSF of the noise and Suu is the PSF of the original signal.
% 

h_psf = otf2psf(h,1*size(flower_blur));
flower_wiener = deconvwnr(flower_blur,h_psf);
figure(7);
subplot(1,3,1),imshow(flower),title('Original image');
subplot(1,3,2),imshow(flower_blur),title('Blurred image');
subplot(1,3,3),imshow(flower_wiener),title('Wiener filtering restored');
err_wiener = immse(flower,flower_wiener)

%%
% 
%  As the result shows, the Wiener filter works just fine as the inverse filter when there's noise.
% 

flower2_wiener = deconvwnr(flower2, h_psf, 0.01/var(im2double(flower(:))));
figure(8);
subplot(2,2,1), imshow(flower2), title('Noisy image');
subplot(2,2,2), imshow(flower2_inverse_uint8), title('Inverse filtering');
subplot(2,2,3), imshow(flower2_pseudo_uint8), title('Pseudo-inverse filtering');
subplot(2,2,4), imshow(flower2_wiener), title('Wiener filtering');
% calculate MSE
err_weiner2 = immse(flower, flower2_wiener)

%%
% 
%  When there's noise in the image, the Wiener filter works better than the inverse filter, while doesn't do as well as the pseudo-inverse filter as it results in a larger MSE.
% 


%% IV. Image Restoration - Blind Convolution

%%
% 
% * Blind deconvolution is a deconvolution technique that permits recovery
% of the target scene from a single or set of "blurred" images in the
% presence of a poorly determined or unknown PSF. It can be performed
% iteratively, whereby each iteration improves the estimation of the PSF
% and the scene, or non-iteratively, where one application of the
% algorithm, based on exterior information, extracts the PSF. Iterative
% methods include maximum a posteriori estimation and
% expectation-maximization algorithms. The PSF of H and the restored image
% are both adjusted in each iteration to reach a better result for the
% given criteria. [5]
% 

%%
% 
%  In MATLAB, there's a function deconvblind to do it iteratively. The estimate of the PSF of the transfer function H is needed, and we can use other parameters such as number of iterations, dampar to control the process.
% 

caterpillar = imresize(imread('caterpillar.jpg'), 0.5);
% give an initial estimate of the psf function
psf = fspecial('motion',13,75);
% control noise amplification
[m, n, o] = size(caterpillar);
dampar = uint8(ones(m, n, o) * 8);
[caterpillar_res,psfr] = deconvblind(caterpillar, psf, 20, dampar);
figure(9)
subplot(1,2,1), imshow(caterpillar), title('Original image');
subplot(1,2,2), imshow(caterpillar_res), title('Blind convoluted');
%%
% 
%  The estimated PSF is a motion blur with 13 pixels and 75-degree counter-clockwise. It gives a fairly decent result, while leaves restortion on the boundary.
% 

%% References

%%
% 
% *[1]* ¡°10.2.6.¡± Digital Image Processing, by Rafael C. Gonzalez and Richard E. Woods, Prentice-Hall, Inc., 2008, pp. 741¨C745.
% 
% *[2]* Mathworkds: Detect and Measure Circular Objects in an Image, from https://www.mathworks.com/help/images/detect-and-measure-circular-objects-in-an-image.html;jsessionid=4552c6fcff7389e22acd529a2dad
% 
% *[3]* Wikipedia: Image restoration, from https://en.wikipedia.org/wiki/Image_restoration
% 
% *[4]* Wikipedia: Wiener filter, from https://en.wikipedia.org/wiki/Wiener_filter
% 
% *[5]* Wikipedia: Blind deconvolution, from https://en.wikipedia.org/wiki/Blind_deconvolution#In_image_processing
% 


