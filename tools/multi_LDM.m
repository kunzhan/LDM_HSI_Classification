function [grade, LDMresult] = multi_LDM(img,train_samples,train_labels,no_classes,P)
lambda1 = P.lambda1;
lambda2 = P.lambda2;
C = P.C;
testInstance = img';
vote = zeros(no_classes,size(testInstance,1));
for i = 1:no_classes-1
    for j = i+1:no_classes
        if i>=j
            error('i must be smaller than j');
        end
        trainInstance = [train_samples(train_labels==i,:);...
            train_samples(train_labels==j,:)];
        label = [-1*ones(length(find(train_labels==i)),1);...
            ones(length(find(train_labels==j)),1)];
        [prediction,~,~] = ...
            LDM(label,trainInstance,ones(size(testInstance,1),1),...
            testInstance,C,lambda1,lambda2,'-s 0 -g 0.1 -k 2');
        for k = 1:size(testInstance,1)
            if prediction(k) == -1
                vote(i,k) = vote(i,k)+1;
            else
                vote(j,k) = vote(j,k)+1;
            end
        end
    end
end
[grade, LDMresult] = max(vote,[],1);