%% Gaussian generation


% author: shenghua
% date: 09/08/2017
% usage:
%    center: the center of the gaussian center in pixel in the image
%    xPixels: the number of pixels in X coordinate
%    yPixels: the number of pixels in Y coordinate
%    gaussian = generate_gaussian([100,100],250,250);

function gaussian = generate_gaussian(cntr, xPixels, yPixels)
    gridRange = 10;
    gaussian = zeros(xPixels, yPixels);
    ctrX = cntr(1);
    ctrY = cntr(2);
    xv = linspace(-gridRange, gridRange,xPixels);
    yv = linspace(-gridRange, gridRange,yPixels);
    [xi,yi] = ndgrid(xv, yv);
    center = [xi(ctrX, ctrY), yi(ctrX, ctrY)];
    sigma = gridRange*6/xPixels*2.5/3;
    exponent = -((xi-center(1)).^2 + (yi-center(2)).^2)./(2*sigma^2);
    amplitude = 1 / (sigma^2 * 2*pi);
    gauss = amplitude * exp(exponent);
    gauss = gauss/sum(sum(gauss));
    gaussian = gauss;
%     [m,n] = size(gauss);
%     image = zeros(m,n,3);
%     image(:,:,1) = gauss;
%     image(:,:,2) = gauss;
%     imshow(image,[])    
end
