-----------------------
-- combine all the indicated datasets into combined one.
-- original label is in LSP 14 joints format, this class is going to provide transferred feeding to
-- MPI format. the  pelvis =  (hipL + hipR)/2 ,
-- throat the same as  13,
-- the upper neck is interpolated as  neck + 1/3(head - neck) ,
-- bb is based on the range of x and y,
-- center is based on bb
-- scale is based on the height/200 of the bb.
-- they are for test only, so all sets are set as test portion

local M = {}
local dir = require 'pl.dir'
require 'torchx'
matio = require 'matio'
local utils = dofile('util/utils.lua')

Dataset = torch.class('pose.Dataset', M)     -- pose seems to be a place holder, makes it a child of M
-- dataset class -- get opt.idxRef

local trainRt = 0.9         -- test ratio in whole dataset
local testRt = 1- trainRt
--local ifdbg = true

function Dataset:__init()   -- opt.idxRef.[train, valid, test],
    -- self: size, skeleton , get, pth, image, infor. dt interface
    -- LSP order seq
    --self.nJoints = 14
    --self.accIdxs = {1,2,3,4,5,6,7,8,11,12}      -- only legs and lower arms
    --self.flipRef = {{1,6}, {2,5}, {3,4},
    --                {7,12}, {8,11},{9,10}}
    ---- Pairs of joints for drawing skeleton, {idxJt1, idxJt2, idxClr}
    --self.skeletonRef = {{1,2,1},    {2,3,1},      --  not using the hip part no pelvis joint
    --                    {4,5,2},    {5,6,2},
    --                    {13,14,0},
    --                    {7,8,3},   {8,9,3},
    --                    {10,11,4},   {11,12,4}}

    -- MPI joints seq
    self.nJoints = 16
    self.accIdxs = {1,2,3,4,5,6,11,12,15,16}    -- only wrist and elbow, two legs for accuracy test
    self.flipRef = {{1,6},   {2,5},   {3,4},
                    {11,16}, {12,15}, {13,14}}
    -- Pairs of joints for drawing skeleton, {idxJt1, idxJt2, idxClr}
    self.skeletonRef = {{1,2,1},    {2,3,1},    {3,7,1},    -- rleg, lleg, torso, head, r arm , l arm,
                        {4,5,2},    {4,7,2},    {5,6,2},
                        {7,9,0},    {9,10,0},
                        {13,9,3},   {11,12,3},  {12,13,3},
                        {14,9,4},   {14,15,4},  {15,16,4}}
    print('generating GPM_ds datasets')

    -- list all file names (abs) and joints_gts, cat them together
    -- dtSizes list size of all datasets
    -- trainIdxs and testIdxs,
    -- if random , then validIdxs  separate  train part
    local imgPths = {}
    local tmpPths = {}
    local idxBase = 0
    local idxSt
    local idxsTest = torch.Tensor()
    local idxsTrValid = torch.Tensor()    -- train valid combined
    local idxsTrain = torch.Tensor()
    local idxsValid = torch.Tensor()
    local tmpIdxs = torch.Tensor()
    local dtSizes = {}
    local szTmp
    local nTrainTmp, nTestTmp, nValidTmp
    local joints_gt=torch.Tensor()
    local visibles = torch.Tensor()
    local centers = torch.Tensor()
    local scales = torch.Tensor()
    local lenTorsos = torch.Tensor()
    local joints_gt_tmp, visiblesTmp, sTmp, joints_cur, lenTorsosTmp -- temp centers and scales for one set , current joints of one person
    local cTmp = torch.Tensor()
    local sTmp = torch.Tensor()
    local curDir
    local tmpJtPth
    local Itmp,ht,wd
    self.centers = torch.Tensor()
    self.scales = torch.Tensor()

    --print('idxsTest is ', idxsTest)
    for i,v in ipairs(opt.GPM_ls) do
        curDir = paths.concat(opt.dataDir,v)    -- datasets/SYN/...
        if ifdbg then
            print('curDir is', curDir)
        end
        --print('current dir is', paths.concat(opt.datasetsDir,v) )
        tmpPths = dir.getallfiles(curDir,'*.jpg')   -- perhaps not in correct order?
        if ifdbg then
            print('jpg images gotten are', tmpPths)
        end
        table.sort(tmpPths) -- just a simple sort, !!
        --print('cur file name is',curDir)
        --print("if file exist", paths.filep(paths.concat(curDir,'joints_gt.mat')))
        joints_gt_tmp = matio.load(paths.concat(curDir,'joints_gt.mat'),'joints_gt')
        -- 3 x 14 x 8000
        --print('joints_gt_tmp before permute is', joints_gt_tmp:size())
        joints_gt_tmp = joints_gt_tmp:permute(3,2,1)    -- 3x14x8000 to 8000 x 14 x 3
        visiblesTmp = 1- joints_gt_tmp:select(3, 3)   -- only visible ( N x16)
        joints_gt_tmp = joints_gt_tmp:narrow(3,1,2) -- bias 1
        lenTorsosTmp = (joints_gt_tmp[{ {}, 3, {}}] - joints_gt_tmp[{ {}, 14, {}}]):norm(2, 2) -- right hip minus the left up shoulder  Nx1 norm of torso

        -- initalize joint_MPi_tmp
        --local joints_MPI_tmp = torch.zeros(joints_gt_tmp:size(1),self.nJoints,2)
        --local visiblesMPI = torch.ones(visiblesTmp:size(1), self.nJoints) -- N x16
        local joints_MPI_tmp = joints_gt_tmp
        local visiblesMPI = visiblesTmp


        imgPths = utils.TableConcat(imgPths, tmpPths)   -- all abs paths
        --print('joints_gt_tmp size', joints_gt_tmp:size())
        -- concat together

        -- version one, roughly estimation
        -- get one image, get size, and calculate the center and scale
        -- center be the image center, scale be the height * (1- margin) /200
        --Itmp = image.load(tmpPths[1])
        --if Itmp:nDimension() < 3 then
        --    ht = Itmp:size(1)
        --else
        --    ht = Itmp:size(2)       -- c x h x w
        --end
        --self.centers = self.centers:cat(torch.Tensor({ht/2,ht/2}):repeatTensor(# tmpPths,1),1)
        --self.scales = self.scales:cat(torch.Tensor({ht/200}):repeatTensor(#tmpPths),1)
        ---- scale should be calculated, because different scale, gaussian will be different, but original work doesn't care about it

        -- version 2 , iter gt x calculate the s and c
        cTmp = torch.Tensor(joints_gt_tmp:size(1),2)    -- initialization
        sTmp = torch.Tensor(joints_gt_tmp:size(1))
        for j = 1, joints_gt_tmp:size(1) do
            joints_cur = joints_gt_tmp[j]   -- 16 x 2
            --print('joints_cur size is', joints_cur:size())
            --cTmp[j] = joints_cur:mean(1)    -- can't index empty one, can't use all joints center, if the joints concentrate to one area, maybe the min max center is better
            cTmp[j] = (joints_cur:max(1) + joints_cur:min(1))/2 -- average value of mean and max
            ht = (joints_cur:max(1) - joints_cur:min(1)):max()
            sTmp[j] = ht/200 * opt.marginScale     -- increase joint based scale
        end

        -- concat together
        --joints_gt = joints_gt:cat(joints_gt_tmp:float(),1)
        joints_gt = joints_gt:cat(joints_MPI_tmp:float(),1)
        --visible = visible:cat(visibleTmp:float(),1) -- concat by rows.
        visibles = visibles:cat(visiblesMPI, 1)
        centers = centers:cat(cTmp:float(), 1)
        scales = scales:cat(sTmp:float(),1)
        lenTorsos = lenTorsos:cat(lenTorsosTmp:float(),1)

        nTrainTmp = 0
        nValidTmp = 0
        nTestTmp = #tmpPths

        idxsTrain = idxsTrain:cat(utils.GenVecFromTo(idxBase+1, idxBase + nTrainTmp))   -- going to be empty tensor
        idxBase = idxBase + nTrainTmp
        idxsValid = idxsValid:cat(utils.GenVecFromTo(idxBase+1, idxBase+ nValidTmp))
        idxBase = idxBase + nValidTmp
        idxsTest = idxsTest:cat(utils.GenVecFromTo(idxBase+1, idxBase+ nTestTmp))   -- one D tensor
        idxBase = idxBase + nTestTmp


    end

    if not opt.idxRef then
        opt.idxRef = {}
        opt.idxRef.test = idxsTest
        if not opt.randomValid then
            opt.idxRef.train = idxsTrain
            opt.idxRef.valid = idxsValid
        else
            idxsTrValid = idxsTrain:cat(idxsValid) -- combined tr valid
            local perm = torch.randperm(idxsTrValid:size(1)):long()
            opt.idxRef.train = idxsTrValid:index(1, perm:sub(1,idxsTrain:size(1)))  -- first train size as train index
            opt.idxRef.valid = idxsTrValid:index(1, perm:sub(idxsTrain:size(1)+1, -1))
            -- rest as valid
        end
    end
    --print('gen vec 1 to 10', utils.GenVecFromTo(1,10))
    --print('valid idxs are', opt.idxRef.valid)
    --print('valid size is', opt.idxRef.valid:size())
    self.joints_gt = joints_gt
    self.visibles = visibles
    self.centers = centers
    self.scales = scales
    self.imgPths = imgPths
    self.lenTorsos = lenTorsos
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel(),
                     test=opt.idxRef.test:numel()}
    --print('the center and scale size is', self.centers:size(), self.scales:size())
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
    if ifdbg then
        print('the path is', self:getPath(idx))
    end
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx)   -- return the parts, center, and scale, how to use scale
    --local pts = self.annot.part[idx]:clone()
    --local c = self.annot.center[idx]:clone()
    --local s = self.annot.scale[idx]
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
