function net = cnnff(net, x,flag)

n = numel(net.layers);
net.layers{1}.a{1} = x;
inputmaps = 1;

% extract CNN features
for l = 2 : n   
    % convolutional layer
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : net.layers{l}.outputmaps   %  for each output map
            
            z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
            
            for i = 1 : inputmaps   %  for each input map
                z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
            end
            
            % activation function: ReLU
            net.layers{l}.a{j} = max((z + net.layers{l}.b{j}),0);
        end
        inputmaps = net.layers{l}.outputmaps;
        
        
    % subsampling layer
    elseif strcmp(net.layers{l}.type, 's')
        
        % max pooling
        for j = 1 : inputmaps
            net.layers{l}.mxInd{j}=FindMaxIndex(net.layers{l-1}.a{j},net.layers{l}.scale);
            maxVal=net.layers{l}.mxInd{j}.*net.layers{l-1}.a{j};
            
            z = convn(maxVal, ones(net.layers{l}.scale) , 'valid');
            net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
        end
    end
end

% vectorize the extracted feature
net.fv = [];
for j = 1 : numel(net.layers{n}.a)
    sa = size(net.layers{n}.a{j});
    net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
end

if flag
    % ETI-pooling
    net.maskTI=[];
    maskvec=zeros(size(net.fv,1),1);
    % feature vectors and the corresponding labels after ETI-pooling
    net.fvTI=[];
    net.fvTIlab=[];
    fvTIlab=zeros(6,1);
    % ETI-pooling for each class
    for k=1:6 % 6 expression classes in total
        fvTIlab(:)=0;
        fvTIlab(k)=1;
        
        for j=1:net.nperTI(k)
            
            tempfv=net.fv(:,3*(j-1)+sum(net.npercls(1:k-1))+1:3*j+sum(net.npercls(1:k-1)));
            % element-wise maximization 
            [fvTI,locTI]=max(tempfv');
            
            net.fvTI=[net.fvTI,repmat(fvTI',1,net.TInum)];
            net.fvTIlab=[net.fvTIlab,repmat(fvTIlab,1,net.TInum)];
            % set mask
            for i=1:net.TInum
                maskvec(:)=0;
                maskvec(find(locTI==i))=1;
                net.maskTI=[net.maskTI,maskvec];
            end
            
        end
        % ETI-pooling for the reminders
        if net.reminder(k)
            tempfv=net.fv(:,size(net.fvTI,2)+1:size(net.fvTI,2)+net.reminder(k));
            if net.reminder(k)==1
                net.fvTI=[net.fvTI,tempfv];
                net.fvTIlab=[net.fvTIlab,fvTIlab];
                net.maskTI=[net.maskTI,ones(size(net.fv,1),1)];
            else
                [fvTI,locTI]=max(tempfv');
                net.fvTI=[net.fvTI,repmat(fvTI',1,net.reminder(k))];
                net.fvTIlab=[net.fvTIlab,repmat(fvTIlab,1,net.reminder(k))];
                for i=1:net.reminder(k)
                    maskvec(:)=0;
                    maskvec(find(locTI==i))=1;
                    net.maskTI=[net.maskTI,maskvec];
                end
            end
        end
    end
    
    
end
end
