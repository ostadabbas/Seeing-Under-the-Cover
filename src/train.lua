-- Prepare tensors for saving network output
local validSamples = opt.validIters * opt.validBatch
saved = {idxs = torch.Tensor(validSamples),
         preds = torch.Tensor(validSamples, unpack(ref.predDim)),
            visibles = torch.Tensor(validSamples, ref.nOutChannels),
            lenTorsos = torch.Tensor(validSamples),
         joints_gt = torch.Tensor(validSamples, ref.nOutChannels, 2)      -- not here at first place?
}
if opt.saveInput then saved.input = torch.Tensor(validSamples, unpack(ref.inputDim)) end
if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(validSamples, unpack(ref.outputDim[1])) end
utils = paths.dofile('util/utils.lua')
-- Main processing step
function step(tag)
    local avgLoss, avgAcc = 0.0, 0.0
    local output, err, idx
    local param, gradparam = model:getParameters()  -- all learnable parameters
    local function evalFn(x) return criterion.output, gradparam end
    local nIters

    if tag == 'train' then
        model:training()    -- training mode
        set = 'train'
        if opt.ftNm ~= '' then
            model:evaluate()        -- fix all
            local mdLast = model.modules[#model.modules]    -- only fine tune last module
            mdLast:training()       -- only fine tune last
            print('fine tune last layer')
        end
    else
        model:evaluate()
        if tag == 'predict' then
            print("==> Generating predictions...")
            local setNm
            local nSamples
            if opt.dataset == 'MPI' then
                setNm = 'valid'     -- special treat the MPI as it lack of the test labels
                if opt.nForceIters >0 then
                    nIters = opt.nForceIters    -- forced iterations
                end
                nSamples = opt[setNm .. 'Iters'] * opt.validBatch
            else
                setNm = 'test'
                nSamples = dataset:size(setNm)
            end
            if opt.nForceIters >0 then
                nIters = opt.nForceIters    -- forced iterations
                nSamples = nIters * 1  -- all batch 1 so not multiply
            end

            saved = {idxs = torch.LongTensor(nSamples),
                     preds = torch.Tensor(nSamples, unpack(ref.predDim)), -- N x n_jts x 5 , 5 outlayers
            visibles = torch.Tensor(nSamples, ref.nOutChannels),
            lenTorsos = torch.Tensor(nSamples),
            joints_gt = torch.Tensor(nSamples,ref.nOutChannels, 2)}
            if opt.saveInput then saved.input = torch.Tensor(nSamples, unpack(ref.inputDim)) end
            if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(nSamples, unpack(ref.outputDim[1])) end  -- output dim  nJt x res x res
            set = setNm
        else
            set = 'valid'
        end
    end -- saved a table with  idxs preds, (input, heatmaps )
    local nIters = opt[set .. 'Iters']
    if opt.nForceIters >0 then
        nIters = opt.nForceIters    -- forced iterations
    end
    --print('set name is before run', set)    -- this yields 0?
    --print('the loader[set] number nsamples is', loader[set].nsamples)
    for i,sample in loader[set]:run() do    -- iter through batches, i index, func evaluated once before the loop is entered then set the i and sample value from the loop , each time all the func generator
        xlua.progress(i, nIters)     -- progress bar
        local input, label, indices = unpack(sample)    -- table unpack, test batch = 1
        if opt.GPU ~= -1 then
            -- Convert to CUDA
            input = applyFn(function (x) return x:cuda() end, input)
            label = applyFn(function (x) return x:cuda() end, label)
        end
        -- Do a forward pass and calculate loss
        local output = model:forward(input)
        local err = criterion:forward(output, label)    -- MSE of heat map and labels
        avgLoss = avgLoss + err / nIters
        if tag == 'train' then
            -- Training: Do backpropagation and optimization
            model:zeroGradParameters()
            model:backward(input, criterion:backward(output, label))
            optfn(evalFn, param, optimState)    -- optim method, weights updated
        else    -- valid or predict , first return crt.output, gradparam,
            -- Validation: Get flipped output
            output = applyFn(function (x) return x:clone() end, output)
            local flippedOut = model:forward(flip(input))
            flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut) -- flip the predictions coordinates left to right
            output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut) -- average output and flipped?! hm
            -- Save sample
            local bs = opt[set .. 'Batch']
            local tmpIdx = (i-1) * bs + 1
            local tmpOut = output
            if type(tmpOut) == 'table' then tmpOut = output[#output] end
            if opt.saveInput then saved.input:sub(tmpIdx, tmpIdx+bs-1):copy(input) end
            if opt.saveHeatmaps then saved.heatmaps:sub(tmpIdx, tmpIdx+bs-1):copy(tmpOut) end
            --print('indices are', indices) need long not float ?
            saved.idxs:sub(tmpIdx, tmpIdx+bs-1):copy(indices)   -- sub, that is batch
            saved.preds:sub(tmpIdx, tmpIdx+bs-1):copy(postprocess(set,indices,output))  -- save in the order of joints, output in original format
            --print('save visibles size', saved.visibles:size())
            --print('dataset visibles size', dataset.visibles:size())
            saved.visibles:sub(tmpIdx, tmpIdx+bs-1):copy(dataset.visibles:index(1,indices))
            saved.lenTorsos:sub(tmpIdx, tmpIdx+bs-1):copy(dataset.lenTorsos:index(1,indices))
            saved.joints_gt:sub(tmpIdx, tmpIdx+bs-1):copy(dataset.joints_gt:index(1,indices))   -- not declared
            --saved.joints_gt:sub(tmpIdx,tmpIdx+bs-1):copy()
        end
        -- Calculate accuracy
        avgAcc = avgAcc + accuracy(output, label) / nIters
    end


     --Print and log some useful metrics
    print(string.format("      %s : Loss: %.7f Acc: %.4f"  % {set, avgLoss, avgAcc}))
    if ref.log[set] then
        table.insert(opt.acc[set], avgAcc)
        ref.log[set]:add{
            ['epoch     '] = string.format("%d" % epoch),
            ['loss      '] = string.format("%.6f" % avgLoss),
            ['acc       '] = string.format("%.4f" % avgAcc),
            ['LR        '] = string.format("%g" % optimState.learningRate)
        }
    end

    if  (tag == 'valid' and opt.snapshot ~= 0 and epoch % opt.snapshot == 0) then   -- only valid save
        model:clearState()
        torch.save(paths.concat(opt.save, 'options.t7'), opt)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
        torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
    end

    if tag == 'valid' or tag == 'predict' then
        -- Take a snapshot
        local predFilename = 'preds.h5'
        if tag == 'predict' then predFilename = 'final_' .. predFilename end
        local predFile = hdf5.open(paths.concat(opt.save, predFilename),'w')
        print('DB save the predictions to file', predFilename);
        for k,v in pairs(saved) do predFile:write(k,v) end
        predFile:close()
    end
end

function train() step('train') end
function valid() step('valid') end
function predict() step('predict') end

function demo()
    print('running demo code now')
    model:evaluate()
    for k,v in ipairs(opt.demoIdxs) do
        --local input, label = loadData('test',v)     -- cropped
        local im = dataset:loadImage(v)
        local pts, center, scale = dataset:getPartInfo(v)
        local inp = crop(im, center, scale, 0,256)
        -- forward network, get last hm and regulate,
        local out = model:forward(inp:view(1,3,256,256):cuda())

        -- draw and combine together
        cutorch.synchronize()
        local hm = out[#out][1]:float() -- last output
        print('hm size is', hm:size())
        hm[hm:lt(0)] = 0        -- less than
        -- Get predictions (coordinates predicts ori and transformed )
        local preds_hm, preds_img = getPreds(hm, center, scale) -- 1 x 16 x 2
        --preds[i]:copy(preds_img)

        preds_hm:mul(4) -- Change to input scale
        print('inp size is', inp:size())
        local dispImg = drawOutput(inp, hm, preds_hm[1])    -- hm and preds is croped
        w = image.display{image=dispImg,win=w}
        sys.sleep(3)
    collectgarbage()
    end
end
