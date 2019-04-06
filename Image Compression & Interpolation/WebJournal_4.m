%% Lab4 : Image Compression & Interpolation
% 
% Updated: v1. 2019-4-3
% By Zhongdao Mo (mzd95@terpmail.umd.edu)
% 

%% I. Image Compression - Basic Ideas
% Image compression is a type of data compression applied to digital
% images, to reduce their cost for storage or transmission. Algorithms may
% take advantage of visual perception and the statistical properties of
% image data to provide superior results compared with generic data
% compression methods which are used for other digital data. [1] The
% diagram of a CODEC system is shown below. [2]

img = imread('CODEC.png');
figure;imshow(img),title('CODEC Model');

%% 
% Encoding is to encode the signal to bit stream, which can be used for
% storage and transmission. In data encoding, the compression can be either
% lossy or lossless. Lossless compression reduces bits by identifying and
% eliminating statistical redundency, and no information is lost. Lossy
% compression reduces bits by removing unnecessary or less important
% information. [3]


%% II. Huffman Coding
% A Huffman code is a particular type of optimal prefix code that is
% commonly used for lossless data compression. [4]
% 

% create symbols with probablity of occurrence
symbols = [1, 2, 3, 4, 5];
prob = [0.5, 0.15, 0.15, 0.1, 0.1];
% create Huffman dictionary based on the symbols and probablities
[dict, avg_len] = huffmandict(symbols, prob);
% generate random signal with the pre defined probablities
sig = randsrc(100,1,[symbols; prob]);
% encode signal
comp = huffmanenco(sig,dict);
% decode signal & verify the data
dsig = huffmandeco(comp,dict);
avg_len
isequal(sig,dsig)

% convert original signal to binary & compute the length
binarySig = de2bi(sig);
seqLen = numel(binarySig);
% convert encoded signal to binary & compute the length
binaryComp = de2bi(comp);
encodedLen = numel(binaryComp);
cratio = seqLen/encodedLen
%% 
% The result shows that the average length for the coded signal is 2. The
% compression ratio will fluctuatea around 3/2 = 1.5, and will be 1.5 only
% if the randomly generated sequence has exactly the same probability
% distribution with the pre-defind probablities.



%% III. Run Length Coding
% Run length coding is a form of lossless data compression in which runs of
% data (sequences in which the same data value occurs in many consecutive
% data elements) are stored as a single data value and count. It's most
% useful on data that contains many such runs. [5]

sig = randsrc(100, 1, [[0, 1]; [0.9, 0.1]])';
% encoding
s = []; len = [];
s(1) = sig(1);
len(1) = 1;
ind = 1;
for i = 2:length(sig)
    if sig(i - 1) == sig(i)
        len(ind) = len(ind) + 1;
    else
        ind = ind + 1;
        s(ind) = sig(i);
        len(ind) = 1;
    end
end
%%%%%%%% break large runs into small ones with maximum length of R %%%%%%%%
R = 8;
% maximum # of repetitions
Rl = floor(len/R);
mRl = max(Rl);    
if mRl > 0
    dl = len-R*floor(len/R);
    % space to store repetitions
    rl = zeros(mRl+1,length(len));
    rl(mRl+1,:) = dl;
    for k=1:mRl
        rl(k,:) = R*(Rl>=k);
    end
    % create repetitions of symbols
    S = s(ones(mRl+1,1),:);
    % unroll both matrices
    len = rl(:)';
    s = S(:)';
    % remove zero length phrases
    lnz = len>0;
    len = len(lnz);
    s = s(lnz);
end
len
s

% decoding
sig1 = [];
for i = 1:length(s)
    sig1 = [sig1 s(i)*ones(1, len(i))];
end
isequal(sig,sig1)
%% 
% The len vector stores the # of runs and the s vector stores the symbol.
% The run length coding can be combined with Huffman coding to furthur
% improve the compression ratio, which will be used in section IV and V for
% the impelementation of DCT transform.


%% IV. Block-based DCT on baboonC
% A discrete cosine transform (DCT) is a  Fourier-related transform similar
% to the discrete Fourier transform (DFT) using only real numbers, to
% express a finite sequence of data points in terms of a sum of cosine
% functions oscillating at different frequencies. [6]

%% 
% The DCT CODEC block diagram is shown below: [7]

img = imread('DCT Transform.jpg');
figure;imshow(img),title('DCT Model');

%% 
% In order to control the PSNR, the scale factor QP which equals 50/quality
% factor, and the number of coefficients preserved in the coefficient
% quantization process needs to be adjusted. The adjustment of QP has
% bigger impact on the PSNR, and the adjustment of the number of
% coefficients preserved is used for fine-tuning of the PSNR.


clear;
I = im2double(imread('baboonC.tiff'));
% standard JPEG quantization matrix
qmatrix=[16 11 10 16  24  40  51  61
    12 12 14 19 26 58 60 55
    14 13 16 24 40 57 69 56
    14 17 22 29 51 87 80 62
    18 22 37 56 68 109 103 77
    24 35 55 64 81 104 113 92
    49 64 78 87 103 121 120 101
    72 92 95 98 112 100 103 99];
QP = 5e-4;
 
%% 
% (1) 8x8 DCT transform

% 8x8 2-D DCT coefficients
T = dctmtx(8);
dct = @(block_struct) T * block_struct.data * T';
dctcoe(:,:,1) = blockproc(I(:,:,1), [8 8], dct);
dctcoe(:,:,2) = blockproc(I(:,:,2), [8 8], dct);
dctcoe(:,:,3) = blockproc(I(:,:,3), [8 8], dct);
figure; imshow(dctcoe), title('DCT Coefficients');
% compute the variance of the coefficients
dct = [im2col(dctcoe(:,:,1), [8 8], 'distinct'), im2col(dctcoe(:,:,2), [8 8], 'distinct'), im2col(dctcoe(:,:,3), [8 8], 'distinct')];
dctvar = var(dct');
figure; plot(dctvar), title('Variance of coefficients');
% PSNR is decided by truncation & scale factor
[~,idx] = sort(dctvar,'descend');
dct_trun = zeros(size(dct));
dct_trun(idx(1:55),:) = dct(idx(1:55),:);
dct1 = dct_trun(:,1 : size(dct_trun,2) / 3);
dct2 = dct_trun(:,size(dct_trun,2) / 3 + 1 : size(dct_trun,2)*2 / 3);
dct3 = dct_trun(:,size(dct_trun,2)*2 / 3 + 1: end);
dctcoe1(:,:,1) = col2im(dct1,[8 8],size(I),'distinct');
dctcoe1(:,:,2) = col2im(dct2,[8 8],size(I),'distinct');
dctcoe1(:,:,3) = col2im(dct3,[8 8],size(I),'distinct');
QM = repmat(qmatrix * QP, size(I(:,:,1))/8);
qdct = floor((dctcoe1 + QM/2)./QM) .* QM;
% zig-zag scan
zig = [zigzag(qdct(:,:,1)) zigzag(qdct(:,:,2)) zigzag(qdct(:,:,3))];
% run-length encode
rlcencode = rlc1(zig);
% run-length decode
rlcdecode = derlc(rlcencode);
% zig-zag inverse
[vmax, hmax, ~] = size(qdct);
len = size(rlcdecode,2);
izig1 = rlcdecode(:,1 : len/3);
izig2 = rlcdecode(:,len/3 + 1 : (len/3) * 2);
izig3 = rlcdecode(:,(len/3) * 2 + 1 : end);
idct(:,:,1) = izigzag(izig1,vmax,hmax);
idct(:,:,2) = izigzag(izig2,vmax,hmax);
idct(:,:,3) = izigzag(izig3,vmax,hmax);
% Resconstruct
invdct = @(block_struct) T' * block_struct.data * T;
I2(:,:,1) = blockproc(idct(:,:,1), [8 8], invdct);
I2(:,:,2) = blockproc(idct(:,:,2), [8 8], invdct);
I2(:,:,3) = blockproc(idct(:,:,3), [8 8], invdct);
figure; imshow(I2);

%% 

% Calculate PSNR, average length used per pixel
peaksnr = psnr(I2, I)
len = avglen(rlcencode)
cratio = size(I,1)*size(I,2)*3*8 / (length(rlcencode) * len)
bitpp = (length(rlcencode) * len) / (size(I,1)*size(I,2)*3)


%% 
% (2) 16x16 DCT transform

% 16x16 2-D DCT coefficients
T = dctmtx(16);
dct = @(block_struct) T * block_struct.data * T';
dctcoe(:,:,1) = blockproc(I(:,:,1), [16 16], dct);
dctcoe(:,:,2) = blockproc(I(:,:,2), [16 16], dct);
dctcoe(:,:,3) = blockproc(I(:,:,3), [16 16], dct);
figure; imshow(dctcoe), title('DCT Coefficients');
% compute the variance of the coefficients
dct = [im2col(dctcoe(:,:,1), [16 16], 'distinct'), im2col(dctcoe(:,:,2), [16 16], 'distinct'), im2col(dctcoe(:,:,3), [16 16], 'distinct')];
dctvar = var(dct');
% PSNR is decided by truncation & scale factor
[~,idx] = sort(dctvar,'descend');
dct_trun = zeros(size(dct));
dct_trun(idx(1:217),:) = dct(idx(1:217),:);
dct1 = dct_trun(:,1 : size(dct_trun,2) / 3);
dct2 = dct_trun(:,size(dct_trun,2) / 3 + 1 : size(dct_trun,2)*2 / 3);
dct3 = dct_trun(:,size(dct_trun,2)*2 / 3 + 1: end);
dctcoe1(:,:,1) = col2im(dct1,[16 16],size(I),'distinct');
dctcoe1(:,:,2) = col2im(dct2,[16 16],size(I),'distinct');
dctcoe1(:,:,3) = col2im(dct3,[16 16],size(I),'distinct');
qmatrix1 = interp2(qmatrix,1,'linear');
qmatrix1(16,:) = qmatrix1(15,:);
qmatrix1(:,16) = qmatrix1(:,15);
qmatrix1 = round(qmatrix1);
QM = repmat(qmatrix1 * QP, size(I(:,:,1))/16);
qdct = floor((dctcoe1 + QM/2)./QM) .* QM;
% zig-zag scan
zig = [zigzag(qdct(:,:,1)) zigzag(qdct(:,:,2)) zigzag(qdct(:,:,3))];
% run-length encode
rlcencode = rlc(zig);
% run-length decode
rlcdecode = derlc(rlcencode);
% zig-zag inverse
[vmax, hmax, ~] = size(qdct);
len = size(rlcdecode,2);
izig1 = rlcdecode(:,1 : len/3);
izig2 = rlcdecode(:,len/3 + 1 : (len/3) * 2);
izig3 = rlcdecode(:,(len/3) * 2 + 1 : end);
idct(:,:,1) = izigzag(izig1,vmax,hmax);
idct(:,:,2) = izigzag(izig2,vmax,hmax);
idct(:,:,3) = izigzag(izig3,vmax,hmax);
% Resconstruct
invdct = @(block_struct) T' * block_struct.data * T;
I3(:,:,1) = blockproc(idct(:,:,1), [16 16], invdct);
I3(:,:,2) = blockproc(idct(:,:,2), [16 16], invdct);
I3(:,:,3) = blockproc(idct(:,:,3), [16 16], invdct);
figure; imshow(I3);

%% 

% Calculate PSNR, average length used per pixel
peaksnr = psnr(I3, I)
len = avglen(rlcencode)
cratio = size(I,1)*size(I,2)*3*8 / (length(rlcencode) * len)
bitpp = (length(rlcencode) * len) / (size(I,1)*size(I,2)*3)

%% 
% (3) 32x32 DCT transform

% 32x32 2-D DCT coefficients
T = dctmtx(32);
dct = @(block_struct) T * block_struct.data * T';
dctcoe(:,:,1) = blockproc(I(:,:,1), [32 32], dct);
dctcoe(:,:,2) = blockproc(I(:,:,2), [32 32], dct);
dctcoe(:,:,3) = blockproc(I(:,:,3), [32 32], dct);
figure; imshow(dctcoe), title('DCT Coefficients');
% compute the variance of the coefficients
dct = [im2col(dctcoe(:,:,1), [32 32], 'distinct'), im2col(dctcoe(:,:,2), [32 32], 'distinct'), im2col(dctcoe(:,:,3), [32 32], 'distinct')];
dctvar = var(dct');
% PSNR is decided by truncation & scale factor
[~,idx] = sort(dctvar,'descend');
dct_trun = zeros(size(dct));
dct_trun(idx(1:820),:) = dct(idx(1:820),:);
dct1 = dct_trun(:,1 : size(dct_trun,2) / 3);
dct2 = dct_trun(:,size(dct_trun,2) / 3 + 1 : size(dct_trun,2)*2 / 3);
dct3 = dct_trun(:,size(dct_trun,2)*2 / 3 + 1: end);
dctcoe1(:,:,1) = col2im(dct1,[32 32],size(I),'distinct');
dctcoe1(:,:,2) = col2im(dct2,[32 32],size(I),'distinct');
dctcoe1(:,:,3) = col2im(dct3,[32 32],size(I),'distinct');
qmatrix2 = interp2(qmatrix,3,'linear');
qmatrix2 = round(qmatrix2(1:32,1:32));
QM = repmat(qmatrix2 * QP, size(I(:,:,1))/32);
qdct = floor((dctcoe1 + QM/2)./QM) .* QM;
% zig-zag scan
zig = [zigzag(qdct(:,:,1)) zigzag(qdct(:,:,2)) zigzag(qdct(:,:,3))];
% run-length encode
rlcencode = rlc(zig);
% run-length decode
rlcdecode = derlc(rlcencode);
% zig-zag inverse
[vmax, hmax, ~] = size(qdct);
len = size(rlcdecode,2);
izig1 = rlcdecode(:,1 : len/3);
izig2 = rlcdecode(:,len/3 + 1 : (len/3) * 2);
izig3 = rlcdecode(:,(len/3) * 2 + 1 : end);
idct(:,:,1) = izigzag(izig1,vmax,hmax);
idct(:,:,2) = izigzag(izig2,vmax,hmax);
idct(:,:,3) = izigzag(izig3,vmax,hmax);
% Resconstruct
invdct = @(block_struct) T' * block_struct.data * T;
I4(:,:,1) = blockproc(idct(:,:,1), [32 32], invdct);
I4(:,:,2) = blockproc(idct(:,:,2), [32 32], invdct);
I4(:,:,3) = blockproc(idct(:,:,3), [32 32], invdct);
figure; imshow(I4);

%% 

% Calculate PSNR, average length used per pixel
peaksnr = psnr(I4, I)
len = avglen(rlcencode)
cratio = size(I,1)*size(I,2)*3*8 / (length(rlcencode) * len)
bitpp = (length(rlcencode) * len) / (size(I,1)*size(I,2)*3)


%% 
% The comparison of the results by different block sizes is shown below.

img = imread('baboonC result.png');
figure;imshow(img),title('Comparison of three block sizes');

%% 
% For this image, the QP needs to be set to 5e-4 to reach a PSNR of 35, and
% the corresponding quality factor is 50 / 5e-4 = 100000 which is very
% high. The number of coefficients preserved is also high. Thus, the
% compression ratio is below 1 and the bit per pixel used for the
% compressed image is bigger than 8.

%% 
% For different block sizes, the QP needed to reach the same PSNR is roughly the same. However, the number of coefficients preserved is
% related to the block size by a linear factor - the number of 16x16 block
% size is approximately 4 times the number of 8x8 block size, and the number of 32x32
% block size is approximately 16 times of the number of 8x8 block size. As
% the number of coefficients preserved rises, more information needs to be
% preserved which leads to a lower compression ratio and higher bit per
% pixel used for the compressed image.





%% V. Block-based DCT on moon
% 
 
QP = 0.012;
%% 
% (1) 8x8 DCT transform

I = imread('moon.tif');
I = im2double(I);
% 8x8 2-D DCT coefficients
T = dctmtx(8);
dct = @(block_struct) T * block_struct.data * T';
dctcoe = blockproc(I, [8 8], dct);
figure; imshow(dctcoe), title('DCT Coefficients');
% compute the variance of the coefficients
dct = im2col(dctcoe, [8 8], 'distinct');
dctvar = var(dct');
figure; plot(dctvar), title('Variance of coefficients');
[~,idx] = sort(dctvar,'descend');
dct_trun = zeros(size(dct));
dct_trun(idx(1:13),:) = dct(idx(1:13),:);
dct1 = dct_trun;
dctcoe1 = col2im(dct1,[8 8],size(I),'distinct');
QM = repmat(qmatrix * QP, size(I)/8);
qdct = floor((dctcoe1 + QM/2)./QM) .* QM;
% zig-zag scan
zig = zigzag(qdct);
% run-length encode
rlcencode = rlc(zig);
% run-length decode
rlcdecode = derlc(rlcencode);
% zig-zag inverse
[vmax, hmax] = size(qdct);
izig1 = rlcdecode;
idct = izigzag(izig1,vmax,hmax);
% Resconstruct
invdct = @(block_struct) T' * block_struct.data * T;
I5 = blockproc(idct, [8 8], invdct);
figure; imshow(I5);

%% 

% Calculate PSNR, average length used per pixel
peaksnr = psnr(I5, I)
len = avglen(rlcencode);
cratio = size(I,1)*size(I,2)*8 / (length(rlcencode) * len)
bitpp = (length(rlcencode) * len) / (size(I,1)*size(I,2))


%% 
% (2) 16x16 DCT transform

I1 = padarray(I,[8 192],'post');
% 16x16 2-D DCT coefficients
T = dctmtx(16);
dct = @(block_struct) T * block_struct.data * T';
dctcoe = blockproc(I1, [16 16], dct);
figure; imshow(dctcoe), title('DCT Coefficients');
% compute the variance of the coefficients
dct = im2col(dctcoe, [16 16], 'distinct');
dctvar = var(dct');
[~,idx] = sort(dctvar,'descend');
dct_trun = zeros(size(dct));
dct_trun(idx(1:34),:) = dct(idx(1:34),:);
dct1 = dct_trun;
dctcoe1 = col2im(dct1,[16 16],size(I1),'distinct');
qmatrix1 = interp2(qmatrix,1,'linear');
qmatrix1(16,:) = qmatrix1(15,:);
qmatrix1(:,16) = qmatrix1(:,15);
qmatrix1 = round(qmatrix1);
QM = repmat(qmatrix1 * QP, size(I1)/16);
qdct = floor((dctcoe1 + QM/2)./QM) .* QM;
% zig-zag scan
zig = zigzag(qdct);
% run-length encode
rlcencode = rlc(zig);
% run-length decode
rlcdecode = derlc(rlcencode);
% zig-zag inverse
[vmax, hmax] = size(qdct);
izig1 = rlcdecode;
idct = izigzag(izig1,vmax,hmax);
% Resconstruct
invdct = @(block_struct) T' * block_struct.data * T;
I6 = blockproc(idct, [16 16], invdct);
I6 = I6(1:size(I,1),1:size(I,2));
figure; imshow(I6);

%% 

% Calculate PSNR, average length used per pixel
peaksnr = psnr(I6, I)
len = avglen(rlcencode);
cratio = size(I,1)*size(I,2)*8 / (length(rlcencode) * len)
bitpp = (length(rlcencode) * len) / (size(I,1)*size(I,2))


%% 
% (3) 32x32 DCT transform

% 32x32 2-D DCT coefficients
T = dctmtx(32);
dct = @(block_struct) T * block_struct.data * T';
dctcoe = blockproc(I1, [32 32], dct);
figure; imshow(dctcoe), title('DCT Coefficients');
% compute the variance of the coefficients
dct = im2col(dctcoe, [32 32], 'distinct');
dctvar = var(dct');
[~,idx] = sort(dctvar,'descend');
dct_trun = zeros(size(dct));
dct_trun(idx(1:112),:) = dct(idx(1:112),:);
dct1 = dct_trun;
dctcoe1 = col2im(dct1,[32 32],size(I1),'distinct');
qmatrix2 = interp2(qmatrix,3,'linear');
qmatrix2 = round(qmatrix2(1:32,1:32));
QM = repmat(qmatrix2 * QP, size(I1)/32);
qdct = floor((dctcoe1 + QM/2)./QM) .* QM;
% zig-zag scan
zig = zigzag(qdct);
% run-length encode
rlcencode = rlc(zig);
% run-length decode
rlcdecode = derlc(rlcencode);
% zig-zag inverse
[vmax, hmax] = size(qdct);
izig1 = rlcdecode;
idct = izigzag(izig1,vmax,hmax);
% Resconstruct
invdct = @(block_struct) T' * block_struct.data * T;
I7 = blockproc(idct, [32 32], invdct);
I7 = I7(1:size(I,1),1:size(I,2));
figure; imshow(I7);

%% 

% Calculate PSNR, average length used per pixel
peaksnr = psnr(I7, I)
len = avglen(rlcencode);
cratio = size(I,1)*size(I,2)*8 / (length(rlcencode) * len)
bitpp = (length(rlcencode) * len) / (size(I,1)*size(I,2))

%% 
% The comparison of the results by different block sizes is shown below.

img = imread('moon result.png');
figure;imshow(img),title('Comparison of three block sizes');
%% 
% For this image, the QP needs to be set to 0.012 to reach a PSNR of 35, and
% the corresponding quality factor is 50 / 0.012 = 4167. This number is
% significantly smaller than that in the baboonC image, which is reasonable
% from that this image has much less color information thus needs lower
% quality to reach the same PSNR.

%% 
% The relation between number of coefficients preserved and the block size
% is similar to the former one. However, due to the low # of coefficients
% that are need to be preserved, a much higher compression ratio is reached
% and the bit per pixel needed for the compressed image is as low as around
% 0.9.



%% VI. Image Interpolation
% Image interpolation is to resize an image. When making an image larger,
% there'll be new pixels without original values. Here I implement three
% types of algorithm to decide those new pixels. [8]

%% 
% (1) Nearest neighbour. Every pixel is replaced with the nearest pixel in
% the output. It can preserve sharp details, but also introduce jaggedness.

%% 
% (2) Bilinear. It's the 2-D extension of the linear interpolation. It
% interpolates pixel color values introduces a continuous transition into
% the output. The result is continuous, but the contrast will be reduced.

%% 
% (3) Bicubic. It's the 2-D extension of the cubic interpolation.

Lena = double(rgb2gray(imread('Lena.jpg')));
Lena = Lena(200:400,150:400);
figure; subplot(2,2,1);
imagesc(Lena); colormap gray; axis off; title('Original');

Lena_liner = interp2(Lena, 5, 'linear');
subplot(2,2,2);
imagesc(Lena_liner); colormap gray; axis off; title('Linear');

Lena_nearest = interp2(Lena, 5, 'nearest');
subplot(2,2,3);
imagesc(Lena_nearest); colormap gray; axis off; title('Nearest neighbor');

Lena_cubic = interp2(Lena, 5, 'cubic');
subplot(2,2,4);
imagesc(Lena_cubic); colormap gray; axis off; title('Cubic');

%% 
% The difference is not obvious in this view. In the full size image of the result, it still can be found that
% the nearest neighbor result has jaggedness which are stairlike lines. The difference
% between bilinear and bicubic is very slight due to the specific case I use here.


%% References
%%
% *[1]* Wikipedia: Image compression, from https://en.wikipedia.org/wiki/Image_compression
%%
% *[2]* "8.1.6 Image Compression Models" Digital Image Processing, by Rafael C. Gonzalez and Richard E. Woods, Prentice-Hall, Inc., 2008, pp. 559.
%%
% *[3]* Wikipedia: Lossy compression, from https://en.wikipedia.org/wiki/Lossy_compression
%%
% *[4]* Wikipedia: Huffman coding, from https://en.wikipedia.org/wiki/Huffman_coding
%%
% *[5]* Wikipedia: Run-length coding, from https://en.wikipedia.org/wiki/Run_length_coding
%%
% *[6]* Wikipedia: Discrete cosine transform, from https://en.wikipedia.org/wiki/Discrete_cosine_transform
%%
% *[7]* "8.2.8 Block Transform Coding" Digital Image Processing, by Rafael C. Gonzalez and Richard E. Woods, Prentice-Hall, Inc., 2008, pp. 588-606.
%%
% *[8]* Wikipedia: Image scaling, from https://en.wikipedia.org/wiki/Image_scaling

%% Appendix

%% 
% (1) Forward zig-zag scanning function:
% 
% <include>zigzag.m</include>
%
%% 
% (2) Inverse zig-zag scanning function:
% 
% <include>izigzag.m</include>
%
%% 
% (3) Run-length coding function:
% 
% <include>rlc.m</include>
%
%% 
% (4) Run-length decoding function:
% 
% <include>derlc.m</include>
%
%% 
% (5) Average compressed data length function:
% 
% <include>avglen.m</include>
%
