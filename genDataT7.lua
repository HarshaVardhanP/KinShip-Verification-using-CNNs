require 'torch'
require 'image'
require 'paths'

--dir = '/home/dravicha/16824/project/KinFaceMD/'
--os.execute("mkdir " .. dir)
torch.setdefaulttensortype('torch.FloatTensor')
img_path =  '/home/dravicha/16824/project/KinFaceFD/images/'
ext = 'jpg'

-- code for which pair to consider to make tensors for
name = 'fd'

dir_p = '/home/dravicha/16824/project/KinFaceFD/'

local imgs = torch.Tensor(500, 2, 3, 64, 64):zero()
local labels = torch.Tensor(500, 1):zero()

--read file with pairs data:
fname = name .. '_pairs.txt'
fname = paths.concat(dir_p,fname)
fh, err = io.open(fname, "rb")
if err then print("error opening file") return end

local idx = 1
for line in io.lines(fname) do
	local count = 1
	for word in  string.gmatch(line, "%S+") do
		if count == 1 then
                        if word == '0' then
                   		labels[{{idx}, {}}] = -1
			else
                        	labels[{{idx},{}}] = 1
                        end
		end
		if count == 2 then
			im1_name = string.sub(word, 2, -2)
			im1_tmp = image.load(paths.concat(img_path, im1_name))
			im1 = image.scale(im1_tmp, 64, 64)
			--im1 = im1*255
			--im1:add(-127.5)
		end
		if count == 3 then
			im2_name = string.sub(word, 2, -2)
			im2_tmp = image.load(paths.concat(img_path, im2_name))
			im2 = image.scale(im2_tmp, 64, 64)
		end
		count = count + 1
 	end
	
	local pairimgs = torch.Tensor(2, 3,64, 64):zero() -- placeholder for each pair of images
	
	pairimgs[{{1}, {}}] = im1
        pairimgs[{{2}, {}}] = im2
        imgs[{{idx}, {}}] = pairimgs
	idx = idx + 1
end
-- save images as t7 file:
imgTensorName = name .. '_pair_imgTensors.t7'

torch.save(paths.concat(dir_p,imgTensorName), imgs, 'ascii')

-- save labels as t7 file:
labelTensorName = name .. '_pair_labelTensors.t7'
torch.save(paths.concat(dir_p,labelTensorName), labels,'ascii')
print("after saving labels")











