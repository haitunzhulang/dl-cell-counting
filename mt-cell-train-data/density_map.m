% generate the Gaussians 


[image,binary,features] = simcep;

cellCoords = features.nuclei.coords;
[xNum,yNum,zNum] = size(image);
densityMap = zeros(xNum,yNum);
for i = 1: length(cellCoords)
   cellCenter = cellCoords{i};
   densityMap = densityMap + generate_gaussian(cellCenter,xNum,yNum);
end


figure(2)
subplot(1,2,1)
imshow(image,[])
subplot(1,2,2)
imshow(densityMap,[])

fid = fopen('image.dat','w');
fwrite(fid, image, 'double');
fclose(fid);

fid = fopen('density.dat','w');
fwrite(fid, densityMap, 'double');
fclose(fid);