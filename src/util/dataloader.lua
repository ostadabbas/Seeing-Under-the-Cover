--
--  Original version: Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--  (Modified a bit by Alejandro Newell)
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--  multi thread each call read in pose

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('DataLoader', M)

function DataLoader.create(opt, dataset, ref)
   -- The train and valid loader
    -- a table contains 3 divisions, create, global vars and ref in each thread
   local loaders = {}

   for i, split in ipairs{'train', 'valid'} do  -- only train and valid split?
      if opt[split .. 'Iters'] > 0 then
         loaders[split] = M.DataLoader(opt, dataset, ref, split)
      end
   end

   return loaders
end

function DataLoader:__init(opt, dataset, ref, split)    -- split (test, train... )
    local function preinit()
        --_G.opt = opt
        paths.dofile('dataset/' .. opt.dataset .. '.lua') -- MPI.lua
    end

    local function init()
        _G.opt, _G.dataset, _G.ref, _G.split = opt, dataset, ref, split -- into global
        -- where loadData in.  no opt, no dataset,
        paths.dofile('../ref.lua')  -- all have refs in threads, channels, global var, dataset ...  in threads
    end

    local function main(idx)    -- main func?
        torch.setnumthreads(1)  -- only one
        return dataset:size(split)  -- return for what
    end

    local threads, sizes = Threads(opt.nThreads, preinit, init, main)
    -- all func last call put into the table initres{} (here sizes), main return size
    --print('returned sizes are', sizes)    nThs x nParas
    self.threads = threads
    self.iters = opt[split .. 'Iters']
    self.batchsize = opt[split .. 'Batch']
    self.nsamples = sizes[1][1]
    self.split = split
end

function DataLoader:size()  -- how many iterations, one epoch
    return self.iters
end

function DataLoader:run()   -- only run iters x batchsizes jobs
    local threads = self.threads
    local iters
    if opt.nForceIters>0 then
        iters = opt.nForceIters
    else
        iters = self.iters
    end
    local size =iters * self.batchsize

    local idxs = torch.range(1,self.nsamples)   -- just 1 to N, if empty, what happen?
    print('current split is', self.split)
    print('there are samples,', self.nsamples)
    for i = 2,math.ceil(size/self.nsamples) do
        idxs = idxs:cat(torch.range(1,self.nsamples))   -- 1,2,...n, 1,2,...
    end -- fill idx multiple round according to nsamples
    -- Shuffle indices
    idxs = idxs:index(1,torch.randperm(idxs:size(1)):long())    -- index 1st dim, 0 dim ts could be 0. randperm need positive, this could be troublesome for empty case.
    -- Map indices to training/validation/test split
    idxs = opt.idxRef[self.split]:index(1,idxs:long()):long()

    local n, idx, sample = 0, 1, nil
    print('the iter size is', size)
    local function enqueue()    -- make threads fully occupied
        while idx <= size and threads:acceptsjob() do   -- narrow(dim, ind, size)
            local indices = idxs:narrow(1, idx, math.min(self.batchsize, size - idx + 1))   -- each time idx jump a block
            threads:addjob(
                function(indices)   -- part of the indices
                    local inp,out = _G.loadData(_G.split, indices)  -- load data should already been read in.
                    collectgarbage()
                    return {inp,out,indices}
                end,
                function(_sample_) sample = _sample_ end, indices
            )   -- put indices at the end, what is that. optional arguments, actually, upvalue can do this
            idx = idx + self.batchsize -- increase idx by batch size
        end
    end

    local function loop()   -- keep threads fully occupied, just do one job, so n and sample updated. Sample = {inp, out, indices}
        enqueue()
        if not threads:hasjob() then return nil end -- no job return nil to end it
        threads:dojob() -- next call back or wait(）
        if threads:haserror() then threads:synchronize() end   -- all queue executed, all jobs in queue.
        enqueue()
        n = n + 1
        return n, sample    -- n indicate batches sample =
    end

    return loop -- outside for loop get the lop func, n and samples?
end

return M.DataLoader
