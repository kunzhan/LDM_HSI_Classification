clc,clear,close all
addpath('data');
addpath('tools');
addpath('LDM');  % download at: http://lamda.nju.edu.cn/code_LDM.ashx
%% Data
load IndiaP;
[r, s, d] = size(img);
GroundT = GroundT';
%% Parameters
k = 40;     sigma_s = 200;  sigma_r = 0.3;
LDM_parameter.lambda1 = 1.25e5;
LDM_parameter.lambda2 = 5e4;
LDM_parameter.C       = 1e6;
%% Training Set and Test Set
no_classes = 16;
no_train   = round(1025);
indexes = train_test_random_new(GroundT(2,:),...
          fix(no_train/no_classes),no_train);

train_labels = GroundT(2,indexes);
Train_class_No = zeros(no_classes,1);
for i =1:16
    Train_class_No(i,1) = length(find(train_labels == i));
end
      
train_indexes = GroundT(:,indexes);
test_indexes = GroundT;
test_indexes(:,indexes) = [];
%% Feature Dimension is Reduced from d to k
fimg = fa(im2vector(img), k);
[fimg] = scale_to_01(fimg);
%% Spatial Structure
fimg = reshape(fimg,[r s k]);
for i = 1:k
    fimg(:,:,i) = RF(fimg(:,:,i),sigma_s,sigma_r,3,fimg(:,:,i));
end
%% Multi-LDM classifer
fimg = im2vector(fimg);
fimg = fimg';
fimg = double(fimg);

train_samples = fimg(:,train_indexes(1,:))';
train_labels  = train_indexes(2,:)';

test_samples  = fimg(:,test_indexes(1,:))';
test_labels   = test_indexes(2,:)';

[train_samples, M, m] = scale_to_n1p1(train_samples);
fimg = scale_to_n1p1(fimg', M, m)';

[grade, LDMresult] = multi_LDM(fimg, train_samples,...
    train_labels, no_classes, LDM_parameter);
%% evaluate the performance
GroundTest = double(test_labels(:,1));
LDMResultTest = LDMresult(test_indexes(1,:));
[LDMOA,LDMAA,LDMkappa,LDMCA] = confusion(GroundTest,LDMResultTest);
%% Display the result of LDM_FL
Final_label = zeros(1,r*s);
Final_label(GroundT(1,:)) = LDMresult(GroundT(1,:));
Final_label = reshape(Final_label,r,s); 
LDM_FL = label2color(Final_label,'india');
figure,imshow(LDM_FL); 
