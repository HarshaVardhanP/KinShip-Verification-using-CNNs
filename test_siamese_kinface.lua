require 'torch'
require 'image'
require 'cunn'
require 'nngraph'
require 'paths'



-----Load the model-------

model_path = '/home/dravicha/16824/project/KinFaceFD/kf_scratch_linReLU_f5_1000.net'
model = torch.load(model_path)
model_KF = model.KF:cuda()
--model_KF:evaluate()
print(model_KF)

collectgarbage()

-----Load the dataset-------
paths.dofile('loadDataset.lua')
print('loaded Dataset')

----Choose test data-------
accuracy = 0
for i=1, 100 do
	local temp = dataset.testdata[{{i},{}}]
	im = temp:squeeze()
	lbl = dataset.testlabels[{{i},{}}]
	local pred = model_KF:forward(im:cuda())
	print("------------------------------------")
	print(pred)
	local pred_lbl = pred[1]
	if (pred_lbl < 1) then pred_lbl = 1
	else pred_lbl = -1
	end
	print("actual label")
        local act_lbl = lbl[1]
	print(act_lbl[1])
	print("predicted label\n\n")
	print(pred_lbl)
	if act_lbl[1] == pred_lbl then 
		accuracy = accuracy+1
	end
end
print(accuracy)








