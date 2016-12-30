function [fimage] = spatial_feature(img,r,eps)
%SPATIAL_FEATURE Summary of this function goes here
%   Detailed explanation goes here
bands=size(img,3);
fimage = 0*img;
for i=1:bands
    fimage(:,:,i) = RF(img(:,:,i),200,0.3,3,img(:,:,i));
end
