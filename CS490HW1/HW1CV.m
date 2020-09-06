function [scd]=getHSVHist(im,nH,nS,nV)
dbg=1;
if dbg
    im = imread('Q:\MATLAB\Projects\CS490HW1\resources\NWPU-RESISC45\lake\lake_006.jpg');
    nH=8; nS=4; nV=4;
end

im1=rgb2hsv(im);

[h, w, dim]=size(im);
x=reshape(im1,[h*w,3]);
% centroids
c1=[1:nH]*(1/(nH+1));
c2=[1:nS]*(1/(nS+1));
c3=[1:nV]*(1/(nV+1));
kclusters = 64;
[centers,assignments] = vl_kmeans(im1,kclusters);
figure(1);
plot(im1(centers==1,1),im1(centers==1,2),'r.','MarkerSize',12)
hold on
plot(im1(centers==2,1),im1(centers==2,2),'b.','MarkerSize',12)
plot(assignments(:,1),assignments(:,2),'kx','MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Centroids','Location','NW')
title 'Cluster Assignments and Centroids'
hold off

[h1,v1]=hist(x(:,1),c1);
[h2,v2]=hist(x(:,1),c2);
[h3,v3]=hist(x(:,1),c3);
figure(2),imshow(im);
figure(3);
subplot(1,3,1); bar(v1, h1); xlabel('hue');
subplot(1,3,2); bar(v2, h2); xlabel('saturation');
subplot(1,3,3); bar(v3, h3); xlabel('value');

return