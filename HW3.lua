-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-action', 'train', 'train or test')
cmd:option('-smoothing', '', 'smoothing method')
cmd:option('-lm', 'count', 'classifier to use')

-- Hyperparameters
-- ...

function train_mle(X, Y, valid_X, valid_Y, nclasses)
  -- Construct count matrix
  local CM = {}
  for i = 1, X:size(1) do
    prefix = X[i]
    if CM[prefix] == nil then
      CM[prefix] = torch.zeros(1, nclasses)
      word = Y[i]
      CM[prefix][word - 1] += 1
    else
      word = Y[i]
      CM[prefix][word - 1] += 1
    end
  end
end

function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   local X = f:read('train_input'):all():long()
   local Y = f:read('train_output'):all():long()
   local valid_X = f:read('valid_input'):all():long()
   local valid_Y = f:read('valid_output'):all():long()
   local test_X = f:read('test_input'):all():long()
   local test_Y = f:read('test_output'):all():long()
   nclasses = f:read('nclasses'):all():long()[1]

   local W = torch.DoubleTensor(nclasses, nfeatures)
   local b = torch.DoubleTensor(nclasses)


   -- Train.
   if opt.action == 'train' then
     print('Training...')
     if opt.lm == 'count' then
       
     end
   end 

   -- Test.
end

main()
