-----------------------
-- combine all the indicated datasets into combined one.
-- original label is in LSP 14 joints format, this class is going to provide transferred feeding to
-- MPI format. the  pelvis =  (hipL + hipR)/2 ,
-- throat the same as  13,
-- the upper neck is interpolated as  neck + 1/3(head - neck) ,
-- bb is based on the range of x and y,
-- center is based on bb
-- scale is based on the height/200 of the bb.
-- joints_gt from the joints_gt_MPI N x 16 x 2
-- bedIR training and test , set specific person for test.
-- limited by subjs, set valid and test as one.

local M = {}
local dir = require 'pl.dir'
require 'torchx'
matio = require 'matio'
require 'pl'

local utils = dofile('util/utils.lua')
Dataset = torch.class('pose.Dataset', M)     -- pose seems to be a place holder, makes it a child of M (empty)
--local idx_subTest ={7}      -- the idx_sub for test
local if_dbg  = nil

function Dataset:__init()   -- opt.idxRef.[train, valid, test],
    -- self: size, skeleton , get, pth, image, infor. dt interface
    -- MPI joints seq
    self.nJoints = 16
    self.accIdxs = {1,2,3,4,5,6,11,12,15,16}    -- only wrist and elbow, two legs for accuracy test
    self.flipRef = {{1,6},   {2,5},   {3,4},
                    {11,16}, {12,15}, {13,14}}
    self.skeletonRef = {{1,2,1},    {2,3,1},    {3,7,1},    -- rleg, lleg, torso, head, r arm , l arm,
                        {4,5,2},    {4,7,2},    {5,6,2},
                        {7,9,0},    {9,10,0},
                        {13,9,3},   {11,12,3},  {12,13,3},
                        {14,9,4},   {14,15,4},  {15,16,4}}
    print('generating datasetPM dataset')

    -- list all file names (abs) and joints_gts, cat them together
    -- dtSizes list size of all datasets
    -- trainIdxs and testIdxs,
    -- if random , then validIdxs  separate  train part
    local imgPths = {}
    local tmpPths = {}
    local idxBase_train = 0
    local idxBase_test = 0
    local idxsTest = torch.Tensor()
    local idxsTrValid = torch.Tensor()    -- train valid combined
    local idxsTrain = torch.Tensor()
    local idxsValid = torch.Tensor()
    local nTrainTmp, nTestTmp, nValidTmp
    local joints_gt=torch.Tensor()
    local visibles = torch.Tensor()
    local centers = torch.Tensor()
    local scales = torch.Tensor()
    local lenTorsos = torch.Tensor()
    local joints_gt_tmp, visiblesTmp, joints_cur, lenTorsosTmp -- temp centers and scales for one set , current joints of one person
    local cTmp = torch.Tensor()
    local sTmp = torch.Tensor()
    local modalFd
    local Itmp,ht,wd
    self.centers = torch.Tensor()
    self.scales = torch.Tensor()
    local subjLs = dir.getdirectories(opt.dataDir)
    --print('data dir is ', opt.dataDir)
    --print('subjLs is', subjLs)
    table.sort(subjLs)
    for i, subjFd in ipairs(subjLs) do
        local mdNm , jtsNm
        if opt.if_PMRGB then
            mdNm = 'RGB'
            --jtsNm = 'joints_gt.mat'
            jtsNm = 'joints_gt_RGB.mat'
        else
            mdNm = 'IR'
            jtsNm = 'joints_gt_IR.mat'
        end
        modalFd = paths.concat(opt.dataDir, subjFd, mdNm)    -- datasets/SYN/...
        if if_dbg then
            print('processing subjFd', subjFd)
        end
        -- check if joints_gt mat file exist, if continue or give up this subj
        -- get the joints gt for this subj
        if path.exists(path.join(subjFd, jtsNm)) then
            joints_gt_tmp = matio.load(path.join(subjFd,jtsNm),'joints_gt')
            local joints_MPI_tmp, visiblesMPI
            joints_MPI_tmp, visiblesMPI, cTmp, sTmp, lenTorsosTmp = utils.LSP2MPI(joints_gt_tmp, opt.marginScale, self.nJoints)
            -- loop the cover
            for i, covNm in ipairs(opt.coverNms) do
                local covFd = path.join(modalFd, covNm)
                tmpPths = dir.getfiles(covFd, '*.png')
                table.sort(tmpPths)
                if if_dbg then
                    --print('cover name is', covNm)
                    print(('images found in %s'):format(covNm))
                end
                imgPths = utils.TableConcat(imgPths, tmpPths)
                        --joints_gt = joints_gt:cat(joints_gt_tmp:float(),1)
                joints_gt = joints_gt:cat(joints_MPI_tmp:float(),1)
                visibles = visibles:cat(visiblesMPI, 1)
                centers = centers:cat(cTmp:float(), 1)
                scales = scales:cat(sTmp:float(),1)
                lenTorsos = lenTorsos:cat(lenTorsosTmp:float(),1)
                local n_smplTmp = #tmpPths
                -- add index to the train and test
                local idx_subTest = {}
                for i = opt.idx_subTest_PM[1], opt.idx_subTest_PM[2] do -- generate a test index list
                    idx_subTest[i] = i
                end
                local subjId = tonumber(subjFd:match('%d%d%d%d%d'))
                --if utils.inTable(subjId, opt.idx_subTest_PM) then
                if utils.inTable(subjId, idx_subTest) then
                    idxsTest= idxsTest:cat(utils.GenVecFromTo(idxBase_test+1, idxBase_test+n_smplTmp))
                    idxsValid = idxsValid:cat(utils.GenVecFromTo(idxBase_test+1, idxBase_test + n_smplTmp))
                    idxBase_test = idxBase_test + n_smplTmp
                else
                    idxsTrain = idxsTrain:cat(utils.GenVecFromTo(idxBase_train+1, idxBase_train + n_smplTmp))
                    idxBase_train = idxBase_train + n_smplTmp
                end
            end
        end
    end
    if not opt.idxRef then
        opt.idxRef = {}
        opt.idxRef.test = idxsTest
        if not opt.randomValid then
            opt.idxRef.train = idxsTrain    -- if empty
            opt.idxRef.valid = idxsValid
        else
            idxsTrValid = idxsTrain:cat(idxsValid) -- combined tr valid
            local perm = torch.randperm(idxsTrValid:size(1)):long() -- further split
            opt.idxRef.train = idxsTrValid:index(1, perm:sub(1,idxsTrain:size(1)))  -- first train size as train index
            opt.idxRef.valid = idxsTrValid:index(1, perm:sub(idxsTrain:size(1)+1, -1))
            -- rest as valid
        end
    end
    self.joints_gt = joints_gt
    self.visibles = visibles
    self.centers = centers
    self.scales = scales
    self.imgPths = imgPths
    self.lenTorsos = lenTorsos
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel(),
                     test=opt.idxRef.test:numel()}
    -- update opts
    opt.testIters = self.nsamples.test
    opt.testBatch = 1
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)   -- concat images to abs path
    --return paths.concat(opt.dataDir,'images',ffi.string(self.annot.imgname[idx]:char():data()))
    return self.imgPths[idx]    -- return the name
end

function Dataset:loadImage(idx) -- get images
    if if_dbg then
        print('the path is', self:getPath(idx))
    end
    local I = image.load(self:getPath(idx))
    if I:size()[1] ==1 then
        I = torch.repeatTensor(I,3,1,1)
    end
    return I
    --return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx)   -- return the parts, center, and scale, how to use scale
    local pts = self.joints_gt[idx]:clone()
    local c = self.centers[idx]:clone()
    local s = self.scales[idx]  -- augmented scale with augmented
    -- Small adjustment so cropping is less likely to take feet out
    --c[2] = c[2] + 15 * s  -- my cente is going to be different, so save it here
    --s = s * 1.25
    return pts, c, s
end

function Dataset:normalize(idx)
    return self.lenTorsos[idx]
end
--
return M.Dataset
