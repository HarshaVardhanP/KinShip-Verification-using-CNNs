require 'torch'
require 'optim'
require 'pl'
require 'paths'

kf = {}

--local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
-- put the labels for each batch in targets
--local targets = torch.Tensor(opt.batchSize * opt.labelSize * opt.labelSize)

local sampleTimer = torch.Timer()
local dataTimer = torch.Timer()


-- create closure to evaluate f(X) and df/dX
local feval = function(x) 
    -- just in case:
    collectgarbage()
    -- get new parameters
    --if x ~= parameters then
    --   parameters:copy(x)
    --end
    -- reset gradients
    gradParameters:zero()
    -- forward pass -> inputs
    local outputs = model_KF:forward((inputs):cuda())
    --print(outputs:size()) 
    --print('---Forarded--')
    local f = criterion:forward(outputs, targets)
    -- estimate df/dW and backward pass
    local df_smpls = criterion:backward(outputs, targets)
    model_KF:backward(inputs, df_smpls)
    -- penalties (L1 and L2):
 --   if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
       -- locals:
  --     local norm,sign= torch.norm,torch.sign
       -- Loss:
  --     f = f + opt.coefL1 * norm(parameters,1)
  --     f = f + opt.coefL2 * norm(parameters,2)^2/2
       -- Gradients:
  --     gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
 --                                                                                                          end
    -- update confusion
  --  for i = 1,opt.batchSize do
  --     confusion:add(outputs[i], targets[i])
  --  end
                                                                                                           -- return f and df/dX
    return f,gradParameters
end



function kf.train(inputs_all)
  cutorch.synchronize()
  epoch = epoch or 1
  local dataLoadingTime = dataTimer:time().real; sampleTimer:reset(); -- timers
  local dataBatchSize = opt.batchSize
  print('in Kf Train func')
  --print(inputs_all:size())
  inputs = inputs_all[1]:cuda()
  targets = inputs_all[2]:cuda()
   
  --inputs = inputs_all.data:cuda()
  --targets = inputs_all.labels:cuda()

  print(inputs:size())
  print(targets:size())

-- TODO: implemnet the training function
-- Perform SGD step:
  sgdState = { 
    learningRate = opt.learningRate,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
  }
  _,fs = optim.sgd(feval, parameters, sgdState)
  print(('Loss : %.5f'):format(fs[1]))
-- disp progress
--xlua.progress(t, dataset:size())


  batchNumber = batchNumber + 1
  cutorch.synchronize(); collectgarbage();
  print(('Epoch: [%d][%d/%d]\tTime %.3f DataTime %.3f'):format(epoch, batchNumber, opt.epochSize, sampleTimer:time().real, dataLoadingTime))
  dataTimer:reset()

end


return kf


