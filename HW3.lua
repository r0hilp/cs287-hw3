-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-action', 'train', 'train or test')
cmd:option('-lm', 'mle', 'classifier to use')

-- Hyperparameters
cmd:option('-alpha', 0.1, 'laplace smoothing alpha')

function make_count_matrix(X, Y, nclasses)
  -- Construct count matrix
  local CM = {}
  for i = 1, X:size(1) do
    prefix = X[i]
    if CM[prefix] == nil then
      CM[prefix] = torch.zeros(1, nclasses)
      word = Y[i]
      CM[prefix][word - 1] = CM[prefix][word - 1] + 1
    else
      word = Y[i]
      CM[prefix][word - 1] = CM[prefix][word - 1] + 1
    end
  end

  return CM
end

function mle_preds(X, CM, nclasses)
  preds = torch.zeroes(X:size(1), nclasses)
  for i = 1, X:size(1) do
    prefix = X[i]
    -- Return uniform distribution if nil
    if CM[prefix] == nil then
      preds[i]:fill(1/nclasses)
    else
      preds[i] = CM[prefix]:div(CM[prefix]:sum())
    end
  end

  return preds
end

function laplace_preds(X, CM, alpha, nclasses)
  preds = torch.zeroes(X:size(1), nclasses)
  for i = 1, X:size(1) do
    prefix = X[i]
    -- Return uniform distribution if nil
    if CM[prefix] == nil then
      preds[i]:fill(1/nclasses)
    else
      CM[prefix]:add(alpha)
      preds[i] = CM[prefix]:div(CM[prefix]:sum())
    end
  end
  
  return preds
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
   nclasses = f:read('nclasses'):all():long()[1]

   -- local W = torch.DoubleTensor(nclasses, nfeatures)
   -- local b = torch.DoubleTensor(nclasses)


   -- Train.
   if opt.action == 'train' then
     print('Training...')
     CM = make_count_matrix(X, Y, nclasses)
     if opt.lm == 'mle' then
       preds = mle_preds(X, CM, nclasses) 
     elseif opt.lm == 'laplace' then
       alpha = opt.alpha
       preds = laplace_preds(X, CM, alpha, nclasses)
     end
   end 

   -- Test.
end

main()
