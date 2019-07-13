function cnnp=DCMACNNs_TrTt(opts)

addpath ./util;

%% load data

% path of original images and patches
path = 'data/JAFFE/';
imgpath = [path 'ori_img.mat'];
patchpath = [path 'patch.mat'];

% load patch data
load(patchpath);
% load images
load(imgpath);
ttdata{opts.row+1,opts.col}=test_x;
ttlab{opts.row+1,opts.col}=test_y;
trdata{opts.row+1,opts.col}=train_x;
trlab{opts.row+1,opts.col}=train_y;

clear test_x test_y train_x train_y ;


%% build model and initialization

%rand('state',0)

% network structure for the local branch
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 3) %convolution layer
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 18, 'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    };
% parallel channel number for ETI-Pooling
cnn.TInum=3;

% network structure for the holistic branch
cnnorg.layers={
    struct('type', 'i')
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 18, 'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    };
cnnorg.TInum=3;

% weights initialization for the local branch
for i=1:opts.row
    for j=1:opts.col
        cnnp{i,j}=cnnsetup(cnn,trdata{i,j},trlab{i,j});
    end
end
% initialization for the holistic branch
cnnp{i+1,j}=cnnsetup(cnnorg,trdata{i+1,j},trlab{i+1,j});
% the aggregated model
cnnp{i+1,1}.ffb=zeros(size(trlab{1,1},1),1);
cnnp{i+1,1}.sumL=zeros(opts.row,opts.col);
% a mask for salient patch learning (Algorithm 1)
cnnp{i+1,1}.pindex=ones(opts.row,opts.col);


%% CNN model training
cnnp=cnntrain(cnnp, trdata, trlab,ttdata, ttlab, opts);

%% model validation
[er, bad, h] = cnntest(cnnp, ttdata, ttlab, opts);

%plot mean squared error
%figure; plot(cnnp{opts.row+1,1}.rL);
fprintf('er is %d \n',er);
end
