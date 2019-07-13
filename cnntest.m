function [er, bad, h] = cnntest(net, x, y, opts)
%  feedforward

flag=0;
net{opts.row+1,1}.fv=[];
net{opts.row+1,1}.ffW=[];
net{opts.row+1,1}.o=[];
for nrow=1:opts.row+1
    for ncol=1:opts.col
        if nrow==opts.row+1 && ncol~=opts.col
            continue;
        end
        
        % ignore bypassed patches
        if nrow<=opts.row && ~net{opts.row+1,1}.pindex(nrow,ncol)
            continue;
        end
        net{nrow,ncol} = cnnff(net{nrow,ncol}, x{nrow,ncol},flag);
        
        net{opts.row+1,1}.fv=[net{opts.row+1,1}.fv;net{nrow,ncol}.fv];
        net{opts.row+1,1}.ffW=[net{opts.row+1,1}.ffW,net{nrow,ncol}.ffW];
    end
end

% fc output
net{opts.row+1,1}.o=softmax(net{opts.row+1,1}.ffW,net{opts.row+1,1}.fv,net{opts.row+1,1}.ffb);
[~, h] = max(net{opts.row+1,1}.o);
[~, a] = max(y{1,1});
bad = find(h ~= a);

er = numel(bad) / size(y{1,1}, 2);
end
