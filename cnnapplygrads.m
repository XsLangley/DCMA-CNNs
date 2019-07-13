function net = cnnapplygrads(net, opts, nrow, ncol)
% weights updating

for l = 2 : numel(net{nrow,ncol}.layers)
    
    if strcmp(net{nrow,ncol}.layers{l}.type, 'c')
        
        for j = 1 : numel(net{nrow,ncol}.layers{l}.a)
            
            for ii = 1 : numel(net{nrow,ncol}.layers{l - 1}.a)
                
                net{nrow,ncol}.layers{l}.k{ii}{j} = net{nrow,ncol}.layers{l}.k{ii}{j} - opts.alpha * net{nrow,ncol}.layers{l}.dk{ii}{j};
            end
            
            
            net{nrow,ncol}.layers{l}.b{j} = net{nrow,ncol}.layers{l}.b{j} - opts.alpha * net{nrow,ncol}.layers{l}.db{j};
        end
    end
end

net{nrow,ncol}.ffW = net{nrow,ncol}.ffW - opts.alpha * net{nrow,ncol}.dffW;
net{opts.row+1,1}.ffb = net{opts.row+1,1}.ffb - opts.alpha * net{opts.row+1,1}.dffb;
end
