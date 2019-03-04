-- Update dimension references to account for intermediate supervision
-- history: 1. Add random center shift. Shuangjun Liu , 18.5.21

ref.predDim = {dataset.nJoints,5}       -- 5 inter layers
ref.outputDim = {}
criterion = nn.ParallelCriterion()
--dataTrans = paths.dofile('data_transforms.lua')
utils = paths.dofile('utils.lua')

for i = 1,opt.nStack do
    ref.outputDim[i] = {dataset.nJoints, opt.outputRes, opt.outputRes}
    criterion:add(nn[opt.crit .. 'Criterion']())
end

-- Function for data augmentation, randomly samples on a normal distribution
local function rnd(x) return math.max(-2*x,math.min(2*x,torch.randn(1)[1]*x)) end
-- max ( -2x,  min( 2x,  N(0) * x)  scale = 0.25  -0.5 to 0.5 range
-- Code to generate training samples from raw images
function generateSample(set, idx)
    -- returned cropped input and also the gaussian label heatmap
    local img = dataset:loadImage(idx)  -- ori image
    local pts, c, s = dataset:getPartInfo(idx)
    local r = 0

    if set == 'train' then
        -- Scale and rotation augmentation
        s = s * (2 ^ rnd(opt.scale))
        r = rnd(opt.rotate)
        if torch.uniform() <= .6 then r = 0 end
        local cSft = torch.randn(2) * opt.cenShift  -- std 1 to +- std 10
        --print('c and cSft is', c, cSft)
        c = c + cSft        -- random shift
    end
    -- add random center


    local inp = crop(img, c, s, r, opt.inputRes)    -- crop to center
    local out = torch.zeros(dataset.nJoints, opt.outputRes, opt.outputRes)  -- 64 x 64
    for i = 1,dataset.nJoints do
        if pts[i][1] > 1 then -- Checks that there is a ground truth annotation
            drawGaussian(out[i], transform(pts[i], c, s, r, opt.outputRes), opt.hmGauss)
        end -- different scale same sigma..
    end -- draw response or leave it as 0

    if set == 'train' then
        -- Flipping and color augmentation
        if torch.uniform() < .5 then
            inp = flip(inp)
            out = shuffleLR(flip(out))
        end
        inp[1]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[2]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[3]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        --print('noise type is', opt.noiseType)
    end
    if opt.ifDsp then
        inp = image.scale(inp, opt.dspRes, opt.dspRes)
        inp = image.scale(inp, opt.inputRes, opt.inputRes)
    end
    if 0 ==opt.noiseType then       -- train or prediction all effect
        -- keep the same inp
    elseif 1 == opt.noiseType then
        inp = utils.AddNoise(opt.noiseStd)(inp):clamp(0,1)
    end
    if opt.ifCanny then
        inp = utils.cannyTs(inp)
    end
    if opt.ifLap then
        inp = utils.lapTs(inp)
    end

    if opt.ifGaussFt then
        local sigma = opt.hmGauss     -- use opts
        local gauK = image.gaussian(math.ceil(3*sigma)*2+1, sigma)      -- amplitude 1
        inp = image.convolve(inp, gauK,'same')
    end

    return inp,out
end

-- Load in a mini-batch of data, certain set(train, valid), idxs
function loadData(set, idxs)
    -- loop through the patches, read in the data one by one and combined into a batch data.
    if type(idxs) == 'table' then idxs = torch.Tensor(idxs) end
    local nsamples = idxs:size(1)
    local input,label

    for i = 1,nsamples do
        local tmpInput,tmpLabel
        tmpInput,tmpLabel = generateSample(set, idxs[i])
        tmpInput = tmpInput:contiguous():view(1,unpack(tmpInput:size():totable()))   -- regulate shape, add batch dimension
        tmpLabel = tmpLabel:view(1,unpack(tmpLabel:size():totable()))
        if not input then
            input = tmpInput
            label = tmpLabel
        else
            input = input:cat(tmpInput,1)
            label = label:cat(tmpLabel,1)
        end
    end

    if opt.nStack > 1 then
        -- Set up label for intermediate supervision
        local newLabel = {}
        for i = 1,opt.nStack do newLabel[i] = label end
        return input,newLabel
    else
        return input,label
    end
end

function postprocess(set, idx, output)
    -- score weighted interpolation -- predicts back to original space
    -- from dataset get information and set it in place
    -- p  original x, y  cropped x,y then scores
    local tmpOutput
    if type(output) == 'table' then tmpOutput = output[#output] -- very last one? stg x batch
    else tmpOutput = output end
    local p = getPreds(tmpOutput)   -- batch x  nJt  positions
    local scores = torch.zeros(p:size(1),p:size(2),1)

    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1,p:size(1) do
        for j = 1,p:size(2) do
            local hm = tmpOutput[i][j]
            local pX,pY = p[i][j][1], p[i][j][2]
            scores[i][j] = hm[pY][pX]
            if pX > 1 and pX < opt.outputRes and pY > 1 and pY < opt.outputRes then
               local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})    -- x + x_diff*0.25
               p[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    p:add(0.5)

    -- Transform predictions back to original coordinate space
    local p_tf = torch.zeros(p:size())
    for i = 1,p:size(1) do
        _,c,s = dataset:getPartInfo(idx[i])
        p_tf[i]:copy(transformPreds(p[i], c, s, opt.outputRes))
    end
    
    return p_tf:cat(p,3):cat(scores,3)
end

function accuracy(output,label)
    if type(output) == 'table' then
        return heatmapAccuracy(output[#output],label[#output],nil,dataset.accIdxs)
    else
        return heatmapAccuracy(output,label,nil,dataset.accIdxs)
    end
end
