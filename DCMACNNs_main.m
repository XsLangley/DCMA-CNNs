%% an example of DCMA-CNN
% only collect six images (each correspond to a specific expression class) from JAFFE dataset for demo
% input image size: 60*60, image patch size: 30*30, overlap rate 0.5

clear all;
close all;


%% parameter settings

% image and patch settings
opts.row=3;
opts.col=3;
opts.imgsize=60;
% learning rate
opts.alpha = 0.1;
% batch size
opts.batchsize = 31;
% epochs number
opts.numepochs = 320;
% starting epoch for salient patch learning (Algoritm 1)
opts.bpst = 200;
% period to bypass less-related patch
opts.bpper = 30;
opts.DB='JA';
% period to save the model
opts.sper = 20;

%% training and testing

DCMACNNs_TrTt(opts);



