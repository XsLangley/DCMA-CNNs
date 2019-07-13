function net = cnnbp(net, y, nrow, ncol, opts)
% back propagation

n = numel(net{nrow,ncol}.layers);

% error and loss
net{opts.row+1,1}.e=net{opts.row+1,1}.o-net{nrow,ncol}.fvTIlab;
net{opts.row+1,1}.L = 1/2* sum(net{opts.row+1,1}.e(:) .^ 2) / size(net{opts.row+1,1}.e, 2);

%%  backprop deltas
net{nrow,ncol}.od = net{opts.row+1,1}.e .* (net{opts.row+1,1}.o .* (1 - net{opts.row+1,1}.o));   %  output delta

net{nrow,ncol}.fvTId=net{nrow,ncol}.ffW'*net{nrow,ncol}.od;

net{nrow,ncol}.fvd=net{nrow,ncol}.fvTId.*net{nrow,ncol}.maskTI;

if strcmp(net{nrow,ncol}.layers{n}.type, 'c')
    convFC=net{nrow,ncol}.fv;
    convFC(find(convFC>0))=1;
    net{nrow,ncol}.fvd=net{nrow,ncol}.fvd.*convFC;
end

sa = size(net{nrow,ncol}.layers{n}.a{1});
fvnum = sa(1) * sa(2);

for j = 1 : numel(net{nrow,ncol}.layers{n}.a)
    net{nrow,ncol}.layers{n}.d{j} = reshape(net{nrow,ncol}.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
end

% backprop, from the (n-1) layer
for l = (n - 1) : -1 : 1
    % convolutional layer
    if strcmp(net{nrow,ncol}.layers{l}.type, 'c')
        for j = 1 : numel(net{nrow,ncol}.layers{l}.a)
            
            derivReLU=net{nrow,ncol}.layers{l}.a{j};
            derivReLU(find(derivReLU>0))=1;
            net{nrow,ncol}.layers{l}.derivReLU{j}=derivReLU;
            net{nrow,ncol}.layers{l}.d{j} = derivReLU .* (expand(net{nrow,ncol}.layers{l + 1}.d{j}, [net{nrow,ncol}.layers{l + 1}.scale net{nrow,ncol}.layers{l + 1}.scale 1]).*net{nrow,ncol}.layers{l+1}.mxInd{j});

        end
     % subsampling layer
    elseif strcmp(net{nrow,ncol}.layers{l}.type, 's')
        % residual of the activation value
        for i = 1 : numel(net{nrow,ncol}.layers{l}.a)
            
            z = zeros(size(net{nrow,ncol}.layers{l}.a{1}));
            
            for j = 1 : numel(net{nrow,ncol}.layers{l + 1}.a)
                z = z + convn(net{nrow,ncol}.layers{l + 1}.d{j}, rot180(net{nrow,ncol}.layers{l + 1}.k{i}{j}), 'full');
            end
            net{nrow,ncol}.layers{l}.d{i} = z;
        end
    end
end

%%  calc gradients

for l = 2 : n
    % only conv layers needed to calculate gradients
    if strcmp(net{nrow,ncol}.layers{l}.type, 'c')
        for j = 1 : numel(net{nrow,ncol}.layers{l}.a)
            for i = 1 : numel(net{nrow,ncol}.layers{l - 1}.a)
                % partial derivative of conv kernels
                net{nrow,ncol}.layers{l}.dk{i}{j} = convn(flipall(net{nrow,ncol}.layers{l - 1}.a{i}), net{nrow,ncol}.layers{l}.d{j}, 'valid') / size(net{nrow,ncol}.layers{l}.d{j}, 3);
                
            end
            % partial derivative of conv bias
            net{nrow,ncol}.layers{l}.db{j} = sum(net{nrow,ncol}.layers{l}.d{j}(:)) / size(net{nrow,ncol}.layers{l}.d{j}, 3);
        end
    end
end

% partial derivative of fc weights
net{nrow,ncol}.dffW = net{nrow,ncol}.od * (net{nrow,ncol}.fvTI.*net{nrow,ncol}.maskTI)' / sum(net{nrow,ncol}.nperTI+net{nrow,ncol}.reminder);
% partial derivative of fc bias
net{opts.row+1,1}.dffb = mean(net{nrow,ncol}.od, 2);

function X = rot180(X)
    X = flipdim(flipdim(X, 1), 2);
end
end
