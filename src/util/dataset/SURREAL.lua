-----------------------
-- combine all the indicated datasets into combined one.
-- original label is in LSP 14 joints format, this class is going to provide transferred feeding to
-- MPI format. the  pelvis =  (hipL + hipR)/2 ,
-- throat the same as  13,
-- the upper neck is interpolated as  neck + 1/3(head - neck) ,
-- bb is based on the range of x and y,
-- center is based on bb
-- scale is based on the height/200 of the bb.
-- This is going walk through the SURREAL dataset, and collect all video data, get the frame numbers, then concat them together. Only collect from run2  datasets with 30% overlap training session.
-- vidPths contains all the mp4 paths.
-- frmPthIdxs contains the mp4 idx and also the frm idx,  combine them to achieve the img and als the labeling.

local M = {}
local dir = require 'pl.dir'
require 'torchx'
cv = require 'cv'
require 'cv.videoio'
--paths.dofile('../../opts.lua')
matio = require 'matio'
local utils = dofile('util/utils.lua')

Dataset = torch.class('pose.Dataset', M)     -- pose seems to be a place holder, makes it a child of M
-- dataset class -- get opt.idxRef

local trainRt = 0.9         -- test ratio in whole dataset
local testRt = 1- trainRt
local dsNm = 'data/cmu/train/run1'       -- only generate data from run1
local frStep =  10  -- every 10 frame take sample
--local ifDbg = true
local ifdbg = true
local vidNumUB = 1000  -- upper bound of the video files, there are too many
local function ToMp4Pth(k,v)
    -- v name <sequenceName>_c%04d_info.mat  -10 is the one needed
    return v:sub(1,-10) .. '.mp4'
end

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
    local idxSUR2MPI= torch.Tensor({8,5,2,3,6,9,1,13,16, 16, 21, 19, 17, 18,20,22}):long() -- map surreal to MPI index
    print('generating SYN dataset')

    -- list all file names (abs) and joints_gts, cat them together
    -- dtSizes list size of all datasets
    -- trainIdxs and testIdxs,
    -- if random , then validIdxs  separate  train part
    local imgPths = {}
    local vidPths = {}      -- all video pths
    local frmPthIdxs = torch.Tensor()   -- all frame index (idxVid idxFrm)
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
    local joints_gt_tmp, visiblesTmp, sTmp, joints_cur, lenTorsosTmp, frmPthIdxsTmp -- temp centers and scales for one set , current joints of one person
    local cTmp = torch.Tensor()
    local sTmp = torch.Tensor()
    local curDir
    local tmpJtPth
    local Itmp,ht,wd
    self.centers = torch.Tensor()
    self.scales = torch.Tensor()

    --print('idxsTest is ', idxsTest)
    --for i,v in ipairs(opt.dsLs) do
    curDir = paths.concat(opt.dataDir,dsNm)    -- datasets/SYN/...
    if ifdbg then
        print('curDir is', curDir)
    end
    --print('current dir is', paths.concat(opt.datasetsDir,v) )
    tmpPths = dir.getallfiles(curDir, '*_info.mat')   -- perhaps not in correct order?
    --if ifdbg then
    --    print('_info.mat gotten are', tmpPths)
    --end
    table.sort(tmpPths) -- just a simple sort, !!
    --tmpPths = {table.unpack(tmpPths,1,math.min(vidNumUB,#tmpPths))} -- limit the number -- too many result to unpack
    print('get total info.mat files', #tmpPths)
    --print('cur file name is',curDir)
    --print("if file exist", paths.filep(paths.concat(curDir,'joints_gt.mat')))
    --vidPths = table.foreach(tmpPths,ToMp4Pth)   -- what we get is a string
    --print('vidPths length is', #vidPths)
   -- --print('vidPths is', vidPths)
   -- local subVidPths = {table.unpack(tmpPths , 1, 5)}
   --print(subVidPths)

    --for i,v in ipairs(tmpPths) do
    local numVidPths = math.min(#tmpPths, vidNumUB)
    for i = 1, numVidPths do
        local v = tmpPths[i]
        xlua.progress(i,numVidPths)
        vidPths[i] = v:sub(1,-10) .. '.mp4'
        --joints_gt_tmp = torch.Tensor()
        joints_gt_tmp = matio.load(v,'joints2D')        -- 2 x24 xT
        local nFrms = math.floor(joints_gt_tmp:size(3)/frStep)
        --print('\n joints gt size 3 is', joints_gt_tmp:size())
        --print('\n nFrames we get is', nFrms)
        if nFrms >0 then
            frmPthIdxsTmp = torch.zeros(nFrms,2)        -- T x2

            frmPthIdxsTmp:narrow(2,1,1):copy(torch.Tensor(nFrms):fill(i))    -- current idx
            frmPthIdxsTmp:narrow(2,2,1):copy(utils.GenVecFromTo(1,nFrms)* frStep)
            --if i >5 and ifdbg then    -- dbg purpose
            --    break
            --end
            -- change coord to joints_MPI_tmp
            joints_gt_tmp = joints_gt_tmp:permute(3,2,1)

            joints_gt_tmp = joints_gt_tmp:index(1,torch.linspace(1*frStep, nFrms*frStep,nFrms):long()) -- ds it
            local joints_MPI_tmp = torch.zeros(joints_gt_tmp:size(1),self.nJoints,2)
            local visiblesMPItmp = torch.ones(nFrms, self.nJoints) -- N x16
            joints_MPI_tmp = joints_gt_tmp:index(2,idxSUR2MPI)
            joints_MPI_tmp[{{},{10},{}}] = joints_MPI_tmp[{{},{9},{}}] + (joints_MPI_tmp[{{},{9},{}}] - joints_MPI_tmp[{{},{8},{}}])*3
            -- visibility unavailable, generate a large ones set only
            -- center, scale lenTorsos
            cTmp = torch.Tensor(nFrms,2)    -- initialization
            sTmp = torch.Tensor(nFrms)
            --print('nFrms is', nFrms)
            --print('joints_gt_size 1 is after permute', joints_gt_tmp:size())
            for j = 1, joints_gt_tmp:size(1) do     -- joints_gt_tmp more infor to cover
                joints_cur = joints_gt_tmp[j]   -- 16 x 2
                --print('joints_cur size is', joints_cur:size())
                --cTmp[j] = joints_cur:mean(1)    -- can't index empty one, can't use all joints center, if the joints concentrate to one area, maybe the min max center is better
                --print('cTmp size is', cTmp:size())
                --print("joints_cur size is",  joints_cur:size())
                cTmp[j] = (joints_cur:max(1) + joints_cur:min(1))/2 -- average value of mean and max
                ht = (joints_cur:max(1) - joints_cur:min(1)):max()
                sTmp[j] = ht/200 * opt.marginScale     -- increase joint based scale
            end
            lenTorsosTmp = (joints_MPI_tmp[{ {}, 3, {}}] - joints_MPI_tmp[{ {}, 14, {}}]):norm(2, 2)
            ---- concat together
            joints_gt = joints_gt:cat(joints_MPI_tmp:float(),1)
            centers = centers:cat(cTmp:float(), 1)
            scales = scales:cat(sTmp:float(),1)
            lenTorsos = lenTorsos:cat(lenTorsosTmp:float(),1)
            frmPthIdxs = frmPthIdxs:cat(frmPthIdxsTmp,1)
            visibles= visibles:cat(visiblesMPItmp,1)
        end

    end

    ---- same scale here, equals to enlarging the sigma range
    ---- get idx for train then combine, idxSt , size
    --nTrainTmp = math.floor(opt.trainRt *  #tmpPths)
    --nValidTmp = math.floor(opt.validRt * #tmpPths)
    --nTestTmp = #tmpPths - nTrainTmp - nValidTmp
    --
    --idxsTrain = idxsTrain:cat(utils.GenVecFromTo(idxBase+1, idxBase + nTrainTmp))
    --idxBase = idxBase + nTrainTmp
    --idxsValid = idxsValid:cat(utils.GenVecFromTo(idxBase+1, idxBase+ nValidTmp))
    --idxBase = idxBase + nValidTmp
    --idxsTest = idxsTest:cat(utils.GenVecFromTo(idxBase+1, idxBase+ nTestTmp))
    --idxBase = idxBase + nTestTmp
    local trainEndIdx = math.floor(opt.trainRt * frmPthIdxs:size(1))
    local validEndIdx = trainEndIdx + math.floor(opt.validRt * frmPthIdxs:size(1))
    idxsTrain = utils.GenVecFromTo(1, trainEndIdx)
    idxsValid = utils.GenVecFromTo(trainEndIdx+1 , validEndIdx)
    idxsTest = utils.GenVecFromTo(validEndIdx+1, frmPthIdxs:size(1))

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
    --print('gen vec 1 to 10', utils.GenVecFromTo(1,10))
    --print('valid idxs are', opt.idxRef.valid)
    --print('valid size is', opt.idxRef.valid:size())
    self.joints_gt = joints_gt
    self.visibles = visibles
    self.centers = centers
    self.scales = scales
    --self.imgPths = imgPths
    self.frmPthIdxs = frmPthIdxs
    self.lenTorsos = lenTorsos
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel(),
                     test=opt.idxRef.test:numel()}
    self.vidPths = vidPths

    --opt.testIters = self.nsamples.test
    opt.testIters = 100        -- surreal too big test 1000
    opt.testBatch = 1
end
function Dataset:ReadFromVid(idxVid,idxFrm)
    local vidPth = self.vidPths[idxVid]

    local cap = cv.VideoCapture{filename=vidPth} -- cv in donkey
    if nil == cap then -- get in
        print('videoCap can not be created from ', path)
    end
    cap:set{propId=1, value=idxFrm-1} --CV_CAP_PROP_POS_FRAMES set frame number

    local rgb
    if pcall(function() _, rgb = cap:read{}; rgb = rgb:permute(3, 1, 2):float()/255; rgb = rgb:index(1, torch.LongTensor{3, 2, 1}) end) then
        return rgb      -- normalized 255
    else
        if (opt.verbose) then print('Img not opened ' .. path,'at frame ', t-1) end
        return nil
    end

end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)   -- concat images to abs path
    --return paths.concat(opt.dataDir,'images',ffi.string(self.annot.imgname[idx]:char():data()))
    --return self.imgPths[idx]    -- return the name
    return self.frmPthIdxs[idx] -- 1x2 ts
end

function Dataset:loadImage(idx) -- get images
    --if ifdbg then
    --    print('the path is', self:getPath(idx))
    --end
    local frmPthIdx = self:getPath(idx)
    local rgb = self:ReadFromVid(frmPthIdx[1], frmPthIdx[2])
    return rgb
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
