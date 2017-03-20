function [prediction,accuracy,value]=LDM(label,trainInstance,groundTruth,testInstance,C,lambda1,lambda2,options)
% function [LDMOA,LDMAA,LDMkappa,LDMCA]=LDM(label,trainInstance,GroundT,testInstance,C,lambda1,lambda2,options)
%  LDM implements the LDM algorithm in [1].
%  ========================================================================
%
%  Input:
%  LDM takes 8 input parameters(the first three parameters are necessary,
%  the rest are optional), in this order:
%
%  label: a column binary vector with length trainInstanceNum. Each element
%         is +1 or -1 and the jth element is the label of the jth row
%         vector of trainInstance.
%
%  trainInstance: a matrix with size trainInstanceNum * dimension. Each row
%                 is a training instance vector.
%
%  groundTruth: a column binary vector with length testInstanceNum. Each
%               element is +1 or -1 and the jth element is the label of the
%               jth row vector of testInstance.
%
%  testInstance: a matrix with size testInstanceNum * dimension. Each row 
%                is a testing instance vector.
%
%  C,lambda1,lambda2: trading-off parameters of LDM.
%
%  options: parameters for LDM.
%           -s solver_type: set type of solver (default 0)\n"
%               0 -- Coordinate Descent (dual)\n"
%               1 -- Average Stochastic Gradient Descent (primal)\n"
%           -k kernel_type: set type of kernel function (default 2)\n"
%               0 -- linear: u'*v\n"
%               1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
%               2 -- rbf: exp(-gamma*|u-v|^2)\n"
%               3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
%           -d degree : set degree in polynomial kernel function (default 3)\n"
%           -g gamma : set gamma in polynomial/rbf/sigmoid kernel function (default 1)\n"
%           -c coef0 : set coef0 in polynomial/sigmoid kernel function (default 0)\n"
%           -t times : set the times to scan data for ASGD\n"
%
%  In our paper, all the features of the instances are normalized to [0,1]
%
%  ========================================================================
%
%  Output:
%  prediction: the predicated labels of test instances
%  accuracy: the accuracy of classification
%  value: the predicated values of test instances
%  
%  ========================================================================
%
%  Example:
%       [prediction,accuracy,value]=LDM(label,train,groundTruth,test,C,lambda1,lambda2,'-s 0 -k 2 -g 0.25');
%
%  ========================================================================
%
%  Reference:
%  [1]  T. Zhang and Z.-H. Zhou. Large margin distribution machine. In: Proceedings of the 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'14), New York, NY, 2014, pp.313-322.

if(issparse(trainInstance)==0)
    trainInstance=sparse(trainInstance);
end
if(issparse(testInstance)==0)
    testInstance=sparse(testInstance);
end


model=trainLDM(label,trainInstance',[C*length(label),lambda1,lambda2],options);
% [prediction,accuracy,value]=predictLDM(groundTruth,testInstance',trainInstance',model);
[prediction,accuracy,value]=predictLDM(groundTruth,testInstance',trainInstance',model);
end


