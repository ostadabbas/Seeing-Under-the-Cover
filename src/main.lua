require 'paths'
paths.dofile('ref.lua')     -- Parse command line input and do global variable initialization， get a global ref variable， global dataset
paths.dofile('model.lua')   -- Read in network model
paths.dofile('train.lua')   -- Load up training/testing functions

-- Set up data loader
torch.setnumthreads(1)
local Dataloader = paths.dofile('util/dataloader.lua')
print('creating data loader...')
loader = Dataloader.create(opt, dataset, ref)  -- loader[set] { valid, train, test}
-- wha tis ref here?
-- Initialize logs
ref.log = {}
ref.log.train = Logger(paths.concat(opt.save, 'train.log'), opt.continue)   -- continue true or false
ref.log.valid = Logger(paths.concat(opt.save, 'valid.log'), opt.continue)
--if not opt.ifEval then
--    print('Saving everything to: ' .. opt.save)
--    os.execute('mkdir -p ' .. opt.save)
--end
-- Main training loop
if opt.finalPredictions then
    opt.nEpochs = 0  -- clear the epoches automatically
end
for i=1,opt.nEpochs do  -- nEpochs train additional how much epoches
    print("==> Starting epoch: " .. epoch .. "/" .. (opt.nEpochs + opt.epochNumber - 1))
    if opt.trainIters > 0 then train() end
    if opt.validIters > 0 then valid() end
    epoch = epoch + 1
    collectgarbage()
end

-- Update reference for last epoch
opt.lastEpoch = epoch - 1   -- updated, but not shown in my options.t7?!!

-- Save model, test result don' have to save in predictions I think
model:clearState()
if not opt.finalPredictions then
    torch.save(paths.concat(opt.save,'options.t7'), opt)
    torch.save(paths.concat(opt.save,'optimState.t7'), optimState)
    torch.save(paths.concat(opt.save,'final_model.t7'), model)
end


-- Generate final predictions on validation set
if opt.finalPredictions then
	ref.log = {}
	loader.test = Dataloader(opt, dataset, ref, 'test')
    print('noise type is', opt.noiseType)
	predict()   -- but change the dataset name to 'valid', maybe the test can only evaluated on MPI benchmark
    -- instead just run valid()
    --valid()
end

if opt.ifDemo then
    demo()
end