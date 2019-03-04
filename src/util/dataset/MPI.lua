local M = {}
Dataset = torch.class('pose.Dataset',M)
-- dataset class -- get opt.idxRef
function Dataset:__init()   -- opt.idxRef.[train, valid, test],
    -- self: size, skeleton , get, pth, image, infor. dt interface
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
    print('generating MPI datasets')

    local lenTorsosPCK = torch.Tensor()
    local annot = {}
    local tags = {'index','person','imgname','part','center','scale',
                  'normalize','torsoangle','visible','multi','istrain'}
    local a = hdf5.open(paths.concat(projectDir,'data/mpii/annot.h5'),'r')  -- coming with project
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    a:close()
    annot.index:add(1)  -- tensor type add 1 to start from 1, should be tensor, index is long tensor,  a field is a list
    annot.person:add(1) -- annot[tag] format table, should index with annot['part'] same
    annot.part:add(1)

    -- Index reference
    if not opt.idxRef then
        local allIdxs = torch.range(1,annot.index:size(1))
        opt.idxRef = {}
        opt.idxRef.test = allIdxs[annot.istrain:eq(0)]
        opt.idxRef.train = allIdxs[annot.istrain:eq(1)]

        if not opt.randomValid then
            -- Use same validation set as used in our paper (and same as Tompson et al)
            tmpAnnot = annot.index:cat(annot.person, 2):long()  -- cat latest dimension
            tmpAnnot:add(-1)

            local validAnnot = hdf5.open(paths.concat(projectDir, 'data/mpii/annot/valid.h5'),'r')
            local tmpValid = validAnnot:read('index'):all():cat(validAnnot:read('person'):all(),2):long()   -- each row  ind person  1,1  1st img 1st person,
            -- vali Anno predefined
            opt.idxRef.valid = torch.zeros(tmpValid:size(1))    -- depends on the valid.h5 1000 from test
            opt.nValidImgs = opt.idxRef.valid:size(1)
            opt.idxRef.train = torch.zeros(opt.idxRef.train:size(1) - opt.nValidImgs)

            -- Loop through to get proper index values
            local validCount = 1
            local trainCount = 1
            for i = 1,annot.index:size(1) do    -- train valid get index
                if validCount <= tmpValid:size(1) and tmpAnnot[i]:equal(tmpValid[validCount]) then
                    opt.idxRef.valid[validCount] = i
                    validCount = validCount + 1
                elseif annot.istrain[i] == 1 then
                    opt.idxRef.train[trainCount] = i
                    trainCount = trainCount + 1
                end
            end
        else
            -- Set up random training/validation split
            local perm = torch.randperm(opt.idxRef.train:size(1)):long()
            opt.idxRef.valid = opt.idxRef.train:index(1, perm:sub(1,opt.nValidImgs))
            opt.idxRef.train = opt.idxRef.train:index(1, perm:sub(opt.nValidImgs+1,-1))
        end
        torch.save(opt.save .. '/options.t7', opt)
    end

    self.annot = annot
    self.joints_gt = annot.part:clone():float()  -- copy to make an interface
    self.visibles = annot.visible
    if opt.ifPCK then
        self.lenTorsos = (annot.part[{{},3,{}}] - annot.part[{{},14,{}}]):norm(2,2)
    else
        self.lenTorsos = annot.normalize
    end
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel(),
                     test=opt.idxRef.test:numel()}

    -- For final predictions
    opt.testIters = self.nsamples.test
    opt.testBatch = 1
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)   -- concat images to abs path
    return paths.concat(opt.dataDir,'images',ffi.string(self.annot.imgname[idx]:char():data()))
end

function Dataset:loadImage(idx) -- get images
    --print('self in load Image is ', self)   -- 10 what is wrong
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx)   -- return the parts, center, and scale, how to use scale
    local pts = self.annot.part[idx]:clone()
    local c = self.annot.center[idx]:clone()
    local s = self.annot.scale[idx]
    -- Small adjustment so cropping is less likely to take feet out
    c[2] = c[2] + 15 * s    -- y = y_ori + 15 * s  move center down a little bit
    s = s * 1.25
    return pts, c, s
end

function Dataset:normalize(idx)
    --return self.annot.normalize[idx]
    return self.lenTorsos[idx]
end

return M.Dataset

