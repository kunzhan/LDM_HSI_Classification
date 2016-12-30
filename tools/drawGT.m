% function drawGT
clear
load IndiaP;
[r, s, ~] = size(img);
GT = zeros(r*s,1);
GT(GroundT(:,1)) = GroundT(:,2);
GT = reshape(GT,r,s);  
gt = label2color(GT,'india');
figure,imshow(gt); 