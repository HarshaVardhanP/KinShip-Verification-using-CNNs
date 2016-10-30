require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
-- require 'datasets.scaled_3d'
require 'pl'
require 'paths'
image_utils = require 'image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end


local sanitize = require('sanitize')


----------------------------------------------------------------------
-- parse command-line options
-- TODO: put your path for saving models in "save" 
opt = lapp[[
  -s,--save          (default "/home/16824/project/KinFaceMS64/")      subdirectory to save logs
  --stopepoch        (default 701)
  --saveFreq         (default 700)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -r,--learningRate  (default 0.05)      learning rate
  -b,--batchSize     (default 100)         batch size
  -m,--momentum      (default 0.9)         momentum term of adam
  -t,--threads       (default 2)           number of threads
  -g,--gpu           (default 0)          gpu to run on (default cpu)
  --scale            (default 512)          scale of images to train on
  --epochSize        (default 200)        number of samples per epoch
  --forceDonkeys     (default 0)
  --nDonkeys         (default 2)           number of data loading threads
  --weightDecay      (default 0.0005)        weight decay
  --classnum         (default 40)    
  --classification   (default 1)
]]

if opt.gpu < 0 or opt.gpu > 8 then opt.gpu = false end
print(opt)

opt.loadSize  = opt.scale 
-- TODO: setup the output size 
 opt.labelSize = 32


opt.manualSeed = torch.random(1,10000) 
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

opt.geometry = {3, opt.scale, opt.scale}
opt.outDim =  opt.classnum


local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.01)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


if opt.network == '' then
  ---------------------------------------------------------------------


-----Single Model for Siamese Network------
  model_1=nn.Sequential()

  model_1:add(nn.SpatialConvolution(3,8,5,5))
  model_1:add(nn.ReLU(true))
  model_1:add(nn.SpatialMaxPooling(2,2,2,2))

  model_1:add(nn.SpatialConvolution(8,32,5,5))
  model_1:add(nn.ReLU(true))
  model_1:add(nn.SpatialMaxPooling(2,2,2,2))
  
  model_1:add(nn.SpatialConvolution(32,64,5,5))
  model_1:add(nn.ReLU(true))

  model_1:add(nn.View(64*9*9))
  model_1:add(nn.Linear(64*9*9,320))
  model_1:add(nn.ReLU(true))
  model_1:add(nn.Linear(320,10))
  model_1:add(nn.Linear(10,2))
--------------------------------------------

----Building Siamese Model-----
  siamese_model = nn.ParallelTable()
  siamese_model:add(model_1)
  siamese_model:add(model_1:clone('weight','bias','gradWeight','gradBias'))

  model_KF = nn.Sequential()
  model_KF:add(nn.SplitTable(1))
  model_KF:add(siamese_model)
  model_KF:add(nn.PairwiseDistance(2))

  model_KF:apply(weights_init)

else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_KF = tmp.KF
end

-- print networks
model_KF:cuda()
print('KF network:')
print(model_KF)

-- TODO: loss function
--criterion = nn.CosineEmbeddingCriterion(0.1)
criterion = nn.HingeEmbeddingCriterion(1)
criterion:cuda()
-- TODO: retrieve parameters and gradients
parameters,grad_parameters = model_KF:getParameters()

paths.dofile('loadDataset.lua')
--print(dataset:size())
print('dataset loading done')

paths.dofile('data.lua')
print('donkeys function done')
-- TODO: setup training functions, use fcn_train_cls.lua
paths.dofile('kf_train.lua')


local optimState = {
    learningRate = opt.learningRate,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}


local function train()
   print('\n<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. optimState.learningRate .. ', momentum = ' .. optimState.momentum .. ']')
  
   model_KF:training()
   batchNumber = 0
   for i=1,opt.epochSize do
         print(dataset:size())
         print('Going to donkey thread')
         print(dataset.data:size())
        -- new_dataset = dataset:clone()
         donkeys:addjob(
         function()
            return makeDataKF(dataset)
         end,
         kf.train)
        -- donkeys:addjob(dataset,kf.train)
   end
   donkeys:synchronize()
   cutorch.synchronize()
end


function train_eachEpoch(dataset)
	
    local time = sys.clock()
    --train one epoch of the dataset
    local epoch_err = 0;
    for mini_batch_start = 1, dataset:size(), opt.batchSize do --for each mini-batch
    	local inputs = {}
        local labels = {}
        --create a mini_batch
        for i = mini_batch_start, math.min(mini_batch_start + opt.batchSize - 1, dataset:size()) do 
            local input = dataset[i][1]:clone() -- the tensor containing two images 
            local label = dataset[i][2] -- +/- 1
            table.insert(inputs, input)
            table.insert(labels, label)
        end  


	local func_eval = 
        function(x)
                --update the model parameters (copy x in to parameters)
                if x ~= parameters then
                    parameters:copy(x) 
                end
                grad_parameters:zero() --reset gradients
                local avg_error = 0 -- the average error of all criterion outs
                --evaluate for complete mini_batch
                for i = 1, #inputs do
                    local output = model_KF:forward(inputs[i]:cuda())

			--printing out outputs of module 10:
			if epoch == 700 then
				op1 = model_KF.modules[2].modules[1].modules[10].output
				op2 = model_KF.modules[2].modules[2].modules[10].output
			end

                    local err = criterion:forward(output:cuda(), labels[i]:cuda())
                    avg_error = avg_error + err
                    --estimate dLoss/dW
                    local dloss_dout = criterion:backward(output:cuda(), labels[i]:cuda())
                    model_KF:backward(inputs[i]:cuda(), dloss_dout:cuda())
                end
                grad_parameters:div(#inputs);
                avg_error = avg_error / #inputs;
                return avg_error, grad_parameters
         end
         
         -- Perform SGD step:
         sgdState = {
            learningRate = opt.learningRate,
            learningRateDecay = 0.0,
            momentum = opt.momentum,
            dampening = 0.0,
            weightDecay = opt.weightDecay
         }
         _,fs = optim.sgd(func_eval, parameters, sgdState)
         epoch_err = epoch_err+fs[1]
         --print(('Loss : %.5f'):format(fs[1]))
         --
    end     
    time = sys.clock() - time
    print(('%6f'):format(epoch_err))
--    print("time taken for 1 epoch = " .. (time * 1000) .. "ms, time taken to learn 1 sample = " .. ((time/dataset:size())*1000) .. 'ms')



end



epoch = 1
-- training loop
while epoch<opt.stopepoch do
  -- train/test
  --model_KF:training()
  --kf.train(dataset)
  --print('Epoch : ',epoch)
  train_eachEpoch(dataset)
  
  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, string.format('kf_scratch_%d.net',epoch))
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, { KF = sanitize(model_KF), opt = opt})
  end

  epoch = epoch + 1

  -- plot errors
  if opt.plot  and epoch and epoch % 1 == 0 then
    torch.setdefaulttensortype('torch.FloatTensor')

    if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  end
end
