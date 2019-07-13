function net = cnnsetup(net, x, y)
% CNN initialization

inputmaps = 1;
mapsize = size(squeeze(x(:, :, 1)));

% initialize each layer
for l = 1 : numel(net.layers)
    if strcmp(net.layers{l}.type, 's')
    % initialize subsampling layers
        for j=1:inputmaps
            net.layers{l}.mxInd{j}=zeros(mapsize);
        end
        
        mapsize = mapsize / net.layers{l}.scale;
        assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
        
        for j = 1 : inputmaps
            net.layers{l}.b{j} = 0;
        end
    end
    if strcmp(net.layers{l}.type, 'c')
        % initialize convolutional layers
        mapsize = mapsize - net.layers{l}.kernelsize + 1;
        fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
        % initialize kernels
        for j = 1 : net.layers{l}.outputmaps
            fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
            for i = 1 : inputmaps  %  input map
                net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
            end
            % bias
            net.layers{l}.b{j} = 0;
            
            % a mask for maxpooling
            net.layers{l}.mask{j}=rand(mapsize);
            % index matrix for maxpooling
            net.layers{l}.maskInd{j}=zeros(mapsize);
        end
        
        inputmaps = net.layers{l}.outputmaps;
    end
end

net.fvnum=prod(mapsize)*inputmaps;

onum = size(y, 1);

% weights for fc layer
net.ffW = (rand(onum, net.fvnum) - 0.5) * 2 * sqrt(6 / (onum + net.fvnum));

end
