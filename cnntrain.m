function net = cnntrain(net, x, y, tx, ty, opts)

m = size(x{1,1}, 3);
numbatches = fix(m / opts.batchsize);

% buffer to store loss
net{opts.row+1,1}.rL = [];
% buffer to store error
if ~isfield(net{opts.row+1,1},'er')
    net{opts.row+1,1}.er=[];
end
if ~isfield(net{opts.row+1,1},'bad')
    net{opts.row+1,1}.bad=[];
end
if ~isfield(net{opts.row+1,1},'h')
    net{opts.row+1,1}.h=[];
end

% training flag: 1 for train, and 0 for test
flag=1;

folderpath=[opts.DB '/'];
if isempty(dir(folderpath))
    mkdir(folderpath)
end

% rearrange the input data before training
tempX=[];
tempY=[];
conseq=[];
for i = 1 : opts.numepochs
    if ~mod(i,10)
        opts.alpha=opts.alpha*0.95;
    end
    tic;
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
    
    % stage 1: extract features from all patches
    % stage 2: bypass unrelated patch, only extract features from the retained patches
    
    
    diff=mod(m,opts.batchsize);
    kk = randperm(m);
    kk = kk(1:m-diff);
    for l = 1 : numbatches
        net{opts.row+1,1}.fv=[];
        net{opts.row+1,1}.ffW=[];
        
        % forward
        for nrow=1:opts.row+1
            for ncol=1:opts.col
                
                
                if nrow==opts.row+1 && ncol~=opts.col
                    continue;
                end
                
                % skip unrelated patches (only works in the stage 2)
                if nrow<=opts.row && ~net{opts.row+1,1}.pindex(nrow,ncol)
                    continue;
                end
                
                % rearrange batch data
                tempX = x{nrow,ncol}(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
                tempY = y{nrow,ncol}(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
                conseq=[];
                for expclass=1:size(y{nrow,ncol},1)
                    conseq=[conseq find(tempY(expclass,:)==1)];
                end
                batch_x{nrow,ncol}=tempX(:,:,conseq);
                batch_y{nrow,ncol}=tempY(:,conseq);
                
                net{nrow,ncol}.npercls=sum(batch_y{nrow,ncol}');
                
                % times to ETI-Pooling in this batch for each class
                net{nrow,ncol}.nperTI=fix(net{nrow,ncol}.npercls/net{nrow,ncol}.TInum);
                
                % ETI-Pooling for the rest part
                net{nrow,ncol}.reminder=mod(net{nrow,ncol}.npercls,net{nrow,ncol}.TInum);
                
                % forward for local/global branch
                net{nrow,ncol} = cnnff(net{nrow,ncol}, batch_x{nrow,ncol},flag);
                
                % calculate the Averaged Cumulative Sum (ACS) for each patch
                if i>(opts.bpst-opts.bpper) && nrow~=opts.row+1
                    net{opts.row+1,1}.sumL(nrow,ncol)=(net{opts.row+1,1}.sumL(nrow,ncol)+sum(net{nrow,ncol}.fv(:).^2)/size(net{nrow,ncol}.fv,2))/numbatches;
                end
                
                % store extracted features as input for the fc layer
                net{opts.row+1,1}.fv=[net{opts.row+1,1}.fv;net{nrow,ncol}.fvTI];
                net{opts.row+1,1}.ffW=[net{opts.row+1,1}.ffW,net{nrow,ncol}.ffW];
            end
        end
                
        % bypass unrelated patches (only works in the stage 2)
        if i>=opts.bpst && ~mod(i-opts.bpst-opts.bpper,opts.bpper) && l==numbatches
            [mrow,mcol]=find(net{opts.row+1,1}.sumL==min(net{opts.row+1,1}.sumL(:)));
            net{opts.row+1,1}.sumL(mrow,mcol)=realmax;
            net{opts.row+1,1}.pindex(mrow,mcol)=0;
        end
        
        % fc outputs
        net{opts.row+1,1}.o=softmax(net{opts.row+1,1}.ffW,net{opts.row+1,1}.fv,net{opts.row+1,1}.ffb);
        
        % back-propagation
        for nrow=1:opts.row+1
            for ncol=1:opts.col
                if nrow==opts.row+1 && ncol~=opts.col
                    continue;
                end
                
                % ignore bypassed patches
                if nrow<=opts.row && ~net{opts.row+1,1}.pindex(nrow,ncol)
                    continue;
                end
                
                % gradient calculation
                net= cnnbp(net, batch_y,nrow,ncol, opts);
                
                % weights updating
                net= cnnapplygrads(net, opts, nrow, ncol);
            end
        end
        
        if isempty(net{opts.row+1,1}.rL)
            net{opts.row+1,1}.rL(1) = net{opts.row+1,1}.L;
        end
        
        net{opts.row+1,1}.rL(end + 1) = net{opts.row+1,1}.L;
        
    end
    
    toc;
    
    if mod(i,5)==0
        disp('check');
        [er, bad, h] = cnntest(net, tx, ty, opts);
        disp(er);
        net{opts.row+1,1}.er(end+1)=er;
        net{opts.row+1,1}.bad(end+1,1:size(bad,2))=bad;
        net{opts.row+1,1}.h(end+1,1:size(h,2))=h;
    end
    
    % save models for every opts.sper epochs
    if mod(i,opts.sper)==0
        sfname=['DCMACNNs_' num2str(i) '.mat'];
        save([folderpath sfname],'net');
    end
end



end