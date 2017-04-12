--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
require 'struct'
require 'image'
require 'string'

paths.dofile('loadDataset.lua')
print('Inside Donkey File')
print(dataset.data:size())


-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "../cache"
os.execute('mkdir -p '..cache)
local trainCache = paths.concat(cache, 'trainCache_assignment2.t7')


-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT') or '../logs'
--------------------------------------------------------------------------------------------
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.loadSize}

-- read the codebook (40 * 3)

local codebooktxt = '/scratch/16824/3d/list/codebook_40.txt' 
local codebook = torch.Tensor(40,3)
if type(opt.classification) == 'number' and opt.classification == 1 then 

  local fcode = torch.DiskFile(codebooktxt, 'r')
  for i = 1, 40 do 
    for j = 1, 3 do 
      codebook[{{i},{j}}] = fcode:readFloat()
    end
  end
  fcode:close()
end


local div_num, sub_num
div_num = 127.5
sub_num = -1


local function loadImage(path)
   local input = image.load(path, 3, 'float')
   input = image.scale(input, opt.loadSize, opt.loadSize)
   input = input * 255
   return input
end


local function loadLabel_high(path)
   local input = image.load(path, 3, 'float')
   input = image.scale(input, opt.loadSize, opt.loadSize )
   input = input * 255
   return input
end



function makeData_cls(img, label)
  -- TODO: the input label is a 3-channel real value image, quantize each pixel into classes (1 ~ 40)
  -- resize the label map from a matrix into a long vector
  -- hint: the label should be a vector with dimension of: opt.batchSize * opt.labelSize * opt.labelSize

  --label : [batchSize, 3 , width, height]
  --looping through batch images

  local new_label = torch.FloatTensor(label:size(1),1,opt.labelSize,opt.labelSize)
   for b=1,label:size(1) do
       local each_label = label[{{b},{},{},{}}]
       local sin_label = torch.squeeze(each_label,1)
      -- print(sin_label:size())
       local rescale_label = image.scale(sin_label,opt.labelSize,opt.labelSize)
      -- print(rescale_label:size())
       for w = 1,rescale_label:size(2) do
          for h = 1,rescale_label:size(3) do
             local label_surfNorm = torch.FloatTensor(3) 
             label_surfNorm[{{1}}]=rescale_label[{{1},{w},{h}}]
             label_surfNorm[{{2}}]=rescale_label[{{2},{w},{h}}]
             label_surfNorm[{{3}}]=rescale_label[{{3},{w},{h}}]
             local prod = torch.FloatTensor(40,1):fill(0)
             prod = codebook * label_surfNorm
             local maxC,maxIdx = torch.max(prod,1)
             new_label[{{b},{},{w},{h}}]=maxIdx
          end
       end
   end
   final_label = new_label:reshape(opt.batchSize * opt.labelSize * opt.labelSize)
  --print(final_label:size())
  return {img, final_label}
end


function makeData_cls_pre(img, label)
  -- TODO: almost same as makeData_cls, need to convert img from RGB to BGR for caffe pre-trained model


   local new_label = torch.FloatTensor(label:size(1),1, opt.labelSize, opt.labelSize)
   for b=1,label:size(1) do
       local each_label = label[{{b},{},{},{}}]
       local sin_label = torch.squeeze(each_label,1)
       --print(sin_label:size())
       local rescale_label = image.scale(sin_label,16,16)
       --print(rescale_label:size())
       for w = 1,rescale_label:size(2) do
          for h = 1,rescale_label:size(3) do
             local label_surfNorm = torch.FloatTensor(3)
             label_surfNorm[{{1}}]=rescale_label[{{1},{w},{h}}]
             label_surfNorm[{{2}}]=rescale_label[{{2},{w},{h}}]
             label_surfNorm[{{3}}]=rescale_label[{{3},{w},{h}}]
             local prod = torch.FloatTensor(40,1):fill(0)
             prod = codebook * label_surfNorm
             local maxC,maxIdx = torch.max(prod,1)
             new_label[{{b},{},{w},{h}}]=maxIdx
          end
       end
   end
   --print(new_label:size())
   final_label = new_label:reshape(opt.batchSize * opt.labelSize * opt.labelSize)

   --RGB to BGR Conversion---    
   img_bgr = img:index(2,torch.LongTensor{3,2,1})

   return {img_bgr, final_label}  
end


function makeDataKF()
   print('makeDataKF func')
   print(dataset.data:size())
   imgs = dataset.data
   labels = dataset.labels
   return {imgs, labels}
end


--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
trainHook = function(self, imgpath, lblpath)
   collectgarbage()
   local img = loadImage(imgpath)
   local label = loadLabel_high(lblpath)
   img:add( - 127.5 )
   label:div(div_num)
   label:add(sub_num)

   return img, label

end

--------------------------------------
-- trainLoader
--[[
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {3, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {3, sampleSize[2], sampleSize[2]}
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(opt.data, 'train')},
      loadSize = {3, loadSize[2], loadSize[2]},
      sampleSize = {3, sampleSize[2], sampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end]]--
collectgarbage()



