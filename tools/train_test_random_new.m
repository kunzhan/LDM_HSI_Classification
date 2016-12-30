function [indexes]=train_test_random_new(y,n,nall)
% function to ramdonly select training samples and testing samples 
% from the whole set of ground truth.

K = max(y);
% generate the  training set
indexes = [];
for i = 1:K
    index1 = find(y == i);
    per_index1 = randperm(length(index1));
    if length(index1)/2>n
        indexes = [indexes ;index1(per_index1(1:n))'];
    else
        indexes = [indexes ;index1(per_index1(1:floor(length(index1)/2)))'];
    end
end
indexes = indexes(:);
indexes_all = [1:length(y)];
indexes_all(indexes) = [];
n_new = nall - length(indexes);
per_indexall = randperm(length(indexes_all));
if n_new>0
    indexes_new = indexes_all(per_indexall(1:n_new));
    indexes = [indexes;indexes_new'];
end
indexes = indexes(:);