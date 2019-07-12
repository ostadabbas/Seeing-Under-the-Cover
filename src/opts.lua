if not opt then

projectDir = projectDir or paths.concat(os.getenv('HOME'),'codesPool/pose-hg-train')        -- code project dir
local ifDbg = true

local function parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text(' ---------- General options ------------------------------------')
    cmd:text()
    cmd:option('-expID',       'exp', 'Experiment ID')
    cmd:option('-dataset',     'SLP', 'Dataset choice: mpii | flic | datasetPM, trigger different interfaces, AC2d for test purpose only ')

    cmd:option('-PMlab',     'danaLab', 'subset of PM dataset of different exp locations')       ------ PM related ------------
    cmd:option('-covNmStr',   'uncover cover1 cover2', 'cover condition separated by space, replace converNms later')
    cmd:option('-coverNms',  {
        'uncover',
        'cover1',
        'cover2',
    }, 'cover name list')
    cmd:option('-ftNm',     '',   'fine tune name suffix for experiment, if yes, only last layer is in training')
    cmd:option('-idx_subTest_SLP_str', '91 102', 'override the idx_subTest_SLP, to indicate the start and end number')
    --cmd:option('-idx_subTest_SLP', {91, 102}, 'subject for test set idx_st idx_end')
    cmd:option('-idx_subTrain_SLP_str', '91 102', 'override the idx_subTest_SLP, to indicate the start and end number')
    --cmd:option('-idx_subTrain_SLP', {1, 90}, 'subject idx for train')
    cmd:option('-ifTsSpec',         false, 'if use specific index for testing, only for ac2d at this time')
    cmd:option('-dataDir',  '/home/jun/datasets', 'Data directory')
    --cmd:option('-expDir',   projectDir .. '/exp',  'Experiments directory')
    cmd:option('expDir',     '/home/jun/exp/pose-hg-train', 'Experiment directory')
    cmd:option('-manualSeed',         -1, 'Manually set RNG seed')
    cmd:option('-GPU',                 1, 'Default preferred GPU, if set to -1: no GPU')
    cmd:option('-finalPredictions',false, 'Generate a final set of predictions at the end of training (default no)')
    cmd:option('-nThreads',            8, 'Number of data loading threads')
    cmd:option('-trainRt',            0.8, 'ratio for train')
    cmd:option('-validRt',            0.1, 'ratio for validation')
    cmd:option('-marginScale',        1.2, 'increase scale coeff to cover out of joints body parts, eg, feet and hands')
    cmd:text()
    cmd:text(' ---------- Model options --------------------------------------')
    cmd:text()
    cmd:option('-netType',          'hg', 'Options: hg | hg-stacked')
    cmd:option('-loadModel',      'none', 'Provide full path to a previously trained model')
    cmd:option('-continue',        false, 'Pick up where an experiment left off')
    cmd:option('-branch',         'none', 'Provide a parent expID to branch off, it should includes the datasetNm ')
    cmd:option('-task',           'pose', 'Network task: pose | pose-int')
    cmd:option('-nFeats',            256, 'Number of features in the hourglass')
    cmd:option('-nStack',              8, 'Number of hourglasses to stack')
    cmd:option('-nModules',            1, 'Number of residual modules at each location in the hourglass')
    cmd:text()
    cmd:text(' ---------- Snapshot options -----------------------------------')
    cmd:text()
    cmd:option('-snapshot',            5, 'How often to take a snapshot of the model (0 = never)')
    cmd:option('-saveInput',       false, 'Save input to the network (useful for debugging)')
    cmd:option('-saveHeatmaps',    false, 'Save output heatmaps')
    cmd:text()
    cmd:text(' ---------- Hyperparameter options -----------------------------')
    cmd:text()
    cmd:option('-LR',             2.5e-4, 'Learning rate')
    cmd:option('-LRdecay',           0.0, 'Learning rate decay')
    cmd:option('-momentum',          0.0, 'Momentum')
    cmd:option('-weightDecay',       0.0, 'Weight decay')
    cmd:option('-alpha',            0.99, 'Alpha')
    cmd:option('-epsilon',          1e-8, 'Epsilon')
    cmd:option('-crit',            'MSE', 'Criterion type')
    cmd:option('-optMethod',   'rmsprop', 'Optimization method: rmsprop | sgd | nag | adadelta')
    cmd:option('-threshold',        .001, 'Threshold (on validation accuracy growth) to cut off training early')
    cmd:text()
    cmd:text(' ---------- Training options -----------------------------------')
    cmd:text()
    cmd:option('-nEpochs',           30, 'Total number of epochs to run, def 100') -- default 30
    cmd:option('-lastEpoch',          0,    'Last epoch position, it will be updated ifi saved in previous session')
    cmd:option('-trainIters',       8000, 'Number of train iterations per epoch') -- default 8000
    cmd:option('-trainBatch',          4, 'Mini-batch size, def 6')
    cmd:option('-validIters',       135, 'Number of validation iterations per epoch, overide in some ds to use all set batch 1 such as datasetPM ')
    cmd:option('-validBatch',          1, 'Mini-batch size for validation')
    cmd:option('-nValidImgs',       1000, 'Number of images to use for validation. Only relevant if randomValid is set to true')
    cmd:option('-randomValid',     false, 'Whether or not to use a fixed validation set of 2958 images (same as Tompson et al. 2015)')
    cmd:option('-cenShift',         10,     'the center shift window to give random center. Window will be shifted with a normal distribution')
    cmd:text()
    cmd:text(' ---------- Data options ---------------------------------------')
    cmd:text()
    cmd:option('-inputRes',          256, 'Input image resolution')
    cmd:option('-outputRes',          64, 'Output heatmap resolution')
    cmd:option('-scale',             .25, 'Degree of scale augmentation')
    cmd:option('-rotate',             30, 'Degree of rotation augmentation')
    cmd:option('-hmGauss',             1, 'Heatmap gaussian size')
    cmd:option('-noiseType',           0,  '0 for no noise, 1 for white noise, keep this consistent for both train and test, reflected as appendix in expID')
    cmd:option('-noiseStd',          0.3,  'standard deviation of the noise, image is range from 0 to 1')
    cmd:option('-ifGaussFt',         false,  'if run the median filter on the images, only the first 3 channel')
    cmd:option('-ifEdge',           false,  'if add the edge channel to the input, to do.....')
    cmd:option('-ifCanny',           false,  'if use the canny operation on read in  image')
    cmd:option('-ifLap',             false,   'if use Laplacian edge detection, can not use with ifCanny together ')
    cmd:option('-cannyTh1',          10,      'threshold1 for canny operation')
    cmd:option('-cannyTh2',          30,      'threshold2 for canny operation, only effective if ifCanny is set')
    cmd:option('-ifEval',           false,  'if evaluation to draw pictures')
    cmd:option('-ifDemo',           false,  'if run demo code to show the result of pose estimation')
    cmd:option('-demoIdxs',         {1,2,3}, 'show estimation result from the target index in test data')
    cmd:option('-ifDsp',            false,  'if run down sample operation before evaluation')
    cmd:option('-if_SLPRGB',         false,  'if SLP read in RGB instead of IR')
    cmd:option('-dspRes',          128,    'downSample resolution ')
    cmd:text(' ---------- evaluation options ---------------------------------------')
    cmd:option('-nForceIters',      0,    'force iteration numbers to overload the dataset iteration numbers')
    cmd:option('-evalNm' ,          'eval_1',    'the evaluation of the drawing')
    cmd:option('-ifPCK',            false,        'if use the PCK standard for MPI evaluation, use torso length')

    local opt = cmd:parse(arg or {})
    --opt.expDir = paths.concat(opt.expDir, opt.dataset)  -- /exp/dataset/expID
    -- keep expDir root position
    opt.dataDir = paths.concat(opt.dataDir, opt.dataset)    -- datasets/SYN one level further to dataset
    if opt.dataset == 'datasetPM' or opt.dataset == 'SLP' then
        opt.dataDir = paths.concat(opt.dataDir, opt.PMlab)      -- add lab path to fd
    end

    -- list args parsing s
    local tmptab = {}
    for word in opt.covNmStr:gmatch("%w+") do
        tmptab[#tmptab +1] = word
    end
    opt.coverNms = tmptab
    local tmptab1 = {}
    for word in opt.idx_subTest_SLP_str:gmatch("%w+") do
        tmptab1[#tmptab1 +1] = tonumber(word)
    end
    opt.idx_subTest_SLP = tmptab1
    local tmptab2 = {}
    for word in opt.idx_subTrain_SLP_str:gmatch("%w+") do
        tmptab2[#tmptab2 +1] = tonumber(word)
    end
    opt.idx_subTrain_SLP = tmptab2

    if opt.dataset == 'AC2d' then   -- change expID if the datasets are list version, other 2 ds original expID
        opt.expID = opt.testLs[1]
    elseif opt.dataset == 'GPM_ds' then
        opt.expID = opt.GPM_ls[1]       -- actually a list but only named after first one,
    elseif opt.dataset == 'SLP' or opt.dataset == 'datasetPM' then      -- overide default expID to autoname with cover conditions
        local suffix = ''
        for i,covNm  in ipairs(opt.coverNms) do
            if covNm:match('u') then
                suffix = suffix .. 'u'
            elseif covNm:match('1') then
                suffix = suffix .. '1'
            elseif covNm:match('2') then
                suffix = suffix .. '2'
            end
        end
        local str_1  = 'cov'
        if opt.if_SLPRGB then
            str_1 = str_1 .. 'RGB'          -- if add RGB to indicate what PM read in
        end
        opt.expID = str_1 .. '-' .. suffix
    end

    local expIDtmp
    if opt.finalPredictions and opt.branch ~= 'none' then
        expIDtmp = paths.basename(opt.branch) .. '_' .. opt.expID   -- add branch name before
        if opt.ifDsp then
            expIDtmp = expIDtmp .. '_dsp'       -- down and upsampled, similar to blurness
        end
        if opt.ifTsSpec then
            expIDtmp = expIDtmp .. '_spec'
        end
        if 1 == opt.noiseType then
            expIDtmp = expIDtmp .. '_wns'
        end
        if opt.ifCanny then
            expIDtmp = expIDtmp .. '_canny'
        end
        if opt.ifLap then
            expIDtmp = expIDtmp .. '_lap'
        end
        if opt.ifGaussFt then   -- expID
           expIDtmp = expIDtmp .. '_gauFt'
        end
        if opt.ifPCK and opt.dataset == 'MPI' then
            expIDtmp = expIDtmp .. '_PCK'       -- for MPI pck standard
        end
        if opt.nForceIters >0  then
            expIDtmp = expIDtmp .. string.format('_it%d', opt.nForceIters)
        end
    elseif opt.loadModel ~= 'none' then
        local hyphen
        if opt.finalPredictions then
            hyphen = '_'
        else
            hyphen = '--'       -- fine tune
        end
        local mdBsNm =  paths.basename(opt.loadModel,'t7')
        if opt.ftNm ~= ''  and not opt.finalPredictions then
            mdBsNm = mdBsNm .. '-' .. opt.ftNm
        end
        expIDtmp = mdBsNm .. hyphen .. opt.expID       -- give add the load name here
        if opt.ifDsp then
            expIDtmp = expIDtmp .. '_dsp'       -- down and upsampled, similar to blurness
        end
        if opt.ifTsSpec then
            expIDtmp = expIDtmp .. '_spec'
        end
        if 1 == opt.noiseType then
            expIDtmp = expIDtmp .. '_wns'
        end
        if opt.ifCanny then
            expIDtmp = expIDtmp .. '_canny'
        end
        if opt.ifLap then
            expIDtmp = expIDtmp .. '_lap'
        end
        if opt.ifGaussFt then   -- expID
           expIDtmp = expIDtmp .. '_gauFt'
        end
        if opt.ifPCK and opt.dataset == 'MPI' then
            expIDtmp = expIDtmp .. '_PCK'       -- for MPI pck standard
        end
        if opt.nForceIters >0  then
            expIDtmp = expIDtmp .. string.format('_it%d', opt.nForceIters)
        end
        --opt.save = paths.concat(opt.expDir, opt.dataset, expIDtmp)   -- add branchNm branch nm
    else    -- for training
        expIDtmp = opt.expID
        if opt.ifDsp then
            expIDtmp = expIDtmp .. '_dsp'       -- down and upsampled, similar to blurness
        end
        if 1 == opt.noiseType then -- add suffix to the expID   white noise
           expIDtmp = expIDtmp .. '_wns'
        end
        if opt.ifCanny then
            expIDtmp = expIDtmp .. '_canny'
        end
        if opt.ifLap then
            expIDtmp = expIDtmp .. '_lap'
        end
        if opt.ifGaussFt then   -- expID
           expIDtmp = expIDtmp .. '_gauFt'
        end
        --opt.save = paths.concat(opt.expDir, opt.dataset, expIDtmp)
    end

    if opt.dataset == 'datasetPM' or opt.dataset == 'SLP' then
        opt.save = paths.concat(opt.expDir, opt.dataset, opt.PMlab, expIDtmp)
    else
        opt.save = paths.concat(opt.expDir, opt.dataset, expIDtmp)   -- add branchNm branch nm  expFd/datasetPM/ expIDtmp
    end
    opt.testRt = 1  - opt.trainRt
    return opt
end

-------------------------------------------------------------------------------
-- Process command line options
-------------------------------------------------------------------------------

opt = parse(arg)    -- all parameters into arg structure



if opt.GPU == -1 then
    nnlib = nn
else
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    nnlib = cudnn
    cutorch.setDevice(opt.GPU)
end

if opt.branch ~= 'none' or opt.continue then -- not prediction from previous expID or from the branch
    -- Continuing training from a prior experiment
    -- Figure out which new options have been set
    local setOpts = {}
    for i = 1,#arg do
        if arg[i]:sub(1,1) == '-' then table.insert(setOpts,arg[i]:sub(2,-1)) end
    end
    -- Where to load the previous options/model from
    if opt.branch ~= 'none' then
        opt.load = opt.expDir .. '/' .. opt.branch -- previous record last epoch -- expDir to expDir + dataset?
    else
        --opt.load = opt.expDir .. '/' .. opt.expID
        opt.load = paths.concat(opt.expDir, opt.dataset, opt,expID)
    end
    if ifDbg then
        print('load path is', opt.load)
    end

    -- Keep previous options, except those that were manually set
    if not opt.finalPredictions then
        local opt_ = opt
        opt = torch.load(opt_.load .. '/options.t7')
        print('read in opt from old opt', opt_.load)
        --print('loaded from ', opt_.load .. '/options.t7')
        --print('loaded opt values are', opt)
        opt.save = opt_.save
        opt.load = opt_.load
        opt.continue = opt_.continue
        opt.dataset = opt_.dataset  -- train or continue mainly for this run
        opt.testLs = opt_.testLs
        opt.dsLs = opt_.dsLs    -- old one will leave a datasetList save
        opt.dataDir = opt_.dataDir
        if not opt.lastEpoch then
            opt.lastEpoch = opt_.lastEpoch
        end
        for i = 1,#setOpts do opt[setOpts[i]] = opt_[setOpts[i]] end    -- take new in changes
    end


    epoch = opt.lastEpoch + 1   -- not updated in my previous version ? what is wrong? empty run?
    -- If there's a previous optimState, load that too
    if paths.filep(opt.load .. '/optimState.t7') then
        optimState = torch.load(opt.load .. '/optimState.t7')
        optimState.learningRate = opt.LR
    end
else epoch = 1 end

if opt.finalPredictions or opt.ifDemo then
    opt.nEpochs = 0 -- if predict
end

    -- other wise, take everything as default set instead of read from files.

opt.epochNumber = epoch

-- Track accuracy
opt.acc = {train={}, valid={}}

-- Save options to experiment directory
if not opt.ifEval then
    print('Saving everything to: ' .. opt.save)
    os.execute('mkdir -p ' .. opt.save)
    torch.save(opt.save .. '/options.t7', opt)
end

--print('idx_test_PM is', opt.idx_subTest_SLP)
end

