require 'image'
require 'torch'
require 'paths'

kinface = {}

--- THIS LOADS ALL THE FACE PAIRS TOGETHER! 2000 RELATIONS
--[[
img_ms = '/home/dravicha/16824/project/KinFaceMD/md_pair_imgTensors.t7'
label_ms = '/home/dravicha/16824/project/KinFaceMD/md_pair_labelTensors.t7'
]]--
--[[
img_md = '/home/dravicha/16824/project/KinFaceMD/md_pair_imgTensors.t7'
label_md = '/home/dravicha/16824/project/KinFaceMD/md_pair_labelTensors.t7'
]]--
--[[
img_fs = '/home/dravicha/16824/project/KinFaceFS/fs_pair_imgTensors.t7'
label_fs = '/home/dravicha/16824/project/KinFaceFS/fs_pair_labelTensors.t7'

img_fd = '/home/dravicha/16824/project/KinFaceFD/fd_pair_imgTensors.t7'
label_fd = '/home/dravicha/16824/project/KinFaceFD/fd_pair_labelTensors.t7'
]]--

img_fd = '/home/dravicha/16824/project/KinFaceFS/fs_pair_imgTensors.t7'
label_fd = '/home/dravicha/16824/project/KinFaceFS/fs_pair_labelTensors.t7'

img = img_fd
label = label_fd
function kinface.loadSingleSet(img, label)--,img_md, label_md, img_fs, label_fs, image_fd, label_fd)
   --print(imgT7)
   local imgs = torch.load(img,'ascii')
   local labels = torch.load(label,'ascii')

   cvset = 2 -- set to (1,5) to get different sets for train-test splits
   dataset = {}

   if cvset == 1 then
        dataset.data = imgs[{{101, 500}, {}}]
        dataset.labels = labels[{{101, 500}, {}}]
        dataset.testdata = imgs[{{1, 100}, {}}]
        dataset.testlabels = labels[{{1, 100}, {}}]
   end


   if cvset == 2 then
        dataset.data = torch.cat(imgs[{{1, 100}, {}}], imgs[{{201, 500}, {}}], 1)
        dataset.labels = torch.cat(labels[{{1, 100}, {}}], labels[{{201, 500}, {}}], 1)
	dataset.testdata = imgs[{{101, 200}, {}}]
        dataset.testlabels = labels[{{101, 200}, {}}]

   end


   if cvset == 3 then
        dataset.data = torch.cat(imgs[{{1, 200}, {}}], imgs[{{301, 500}, {}}], 1)
        dataset.labels = torch.cat(labels[{{1, 200}, {}}], labels[{{301, 500}, {}}], 1)
	dataset.testdata = imgs[{{201, 300}, {}}]
        dataset.testlabels = labels[{{201, 300}, {}}]

   end


   if cvset == 4 then
	dataset.data = torch.cat(imgs[{{1, 300}, {}}], imgs[{{401, 500}, {}}], 1)
        dataset.labels = torch.cat(labels[{{1, 300}, {}}], labels[{{401, 500}, {}}], 1)
	dataset.testdata = imgs[{{301, 400}, {}}]
        dataset.testlabels = labels[{{301, 400}, {}}]

   end

   if cvset == 5 then
   	dataset.data = imgs[{{1, 400}, {}}]
   	dataset.labels = labels[{{1, 400}, {}}]
	dataset.testdata = imgs[{{401, 500}, {}}]
        dataset.testlabels = labels[{{401, 500}, {}}]

   end
   print('Size of training data:')
   print(dataset.data:size())
   
   --------Normalization Preprocessing----
--   dataset.data:add(-127.5)
--   dataset.testdata:add(-127.5)
   --[[
   local mean1 = torch.mean(imgs, 1)
   local mean2 = torch.mean(mean1, 2)
   local mean_image = mean2:squeeze()   
   local resized_meanimg = torch.repeatTensor(mean_image,400, 2, 1, 1, 1)
print(resized_meanimg:size())
print(dataset.data:size())
print(dataset.testdata:size())
   dataset.data:add(-resized_meanimg)
   dataset.testdata:add(-torch.repeatTensor(mean_image,100, 2, 1, 1, 1))
]]--
--[[
x = torch.Tensor(4,4)
i = 0

x:apply(function()
  i = i + 1
  return i
end)

   print(x)
local y = torch.repeatTensor(x, 3, 1, 1)
print(y)
local a = torch.repeatTensor(y, 2, 3, 1, 1, 1)
print(a:size())
print(a)   
]]--
--[[
local std = torch.std(imgs)
   dataset.data:mul(1.0/std)
   dataset.testdata:mul(1.0/std)
   dataset.std = std
   dataset.mean = mean_image
   
]]--

   function dataset:size()
      return dataset.data:size(1)
   end

   setmetatable(dataset, {__index = function(self, index)
                        local input = self.data[index]
                        local label = self.labels[index]
                        local example = {input, label}
                        return example
                        end })

   --print(dataset:size())
   return dataset
end

dataset = kinface.loadSingleSet(img, label)--, img_md, label_md, img_fs, label_fs, image_fd, label_fd)


