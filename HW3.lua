-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-action', 'train', 'train or test')
cmd:option('-smoothing', '', 'smoothing method')
cmd:option('-lm', 'mle', 'classifier to use')

cmd:option('-window_size', 5, 'window size')
cmd:option('-warm_start', '', 'torch file with previous model')
cmd:option('-test_model', '', 'model to test on')
cmd:option('-model_out_name', 'train', 'output file name of model')

-- Hyperparameters
cmd:option('-alpha', 0.1, 'laplace smoothing alpha')

cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-batch_size', 32, 'batch size for SGD')
cmd:option('-max_epochs', 20, 'max # of epochs for SGD')
cmd:option('-L2s', 1, 'normalize L2 of word embeddings')

cmd:option('-embed', 50, 'size of word embeddings')
cmd:option('-hidden', 100, 'size of hidden layer for neural network')

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

function NNLM()
  if opt.warm_start ~= '' then
    return torch.load(opt.warm_start).model
  end

  local model = nn.Sequential()
  local word_embed = nn.LookupTable(nfeatures, opt.embed)
  word_embed.weight[1]:zero()
  model:add(word_embed)
  model:add(nn.View(opt.embed * window_size)) -- concat

  model:add(nn.Linear(opt.embed * window_size, opt.hidden))
  model:add(nn.Tanh())
  model:add(nn.Linear(opt.hidden, nclasses))
  model:add(nn.LogSoftMax())
  -- skip connections?
  return model
end

function compute_err(Y, pred)
  -- Compute error from Y
  local _, argmax = torch.max(pred, 2)
  argmax:squeeze()

  local correct
  if Y then
    correct = argmax:eq(Y):sum()
  end
  return argmax, correct
end

function model_eval(model, criterion, X, Y)
    -- batch eval
    model:evaluate()
    local N = X:size(1)
    local batch_size = opt.batch_size

    local total_loss = 0
    local total_correct = 0
    for batch = 1, X:size(1), batch_size do
        local sz = batch_size
        if batch + batch_size > N then
          sz = N - batch + 1
        end
        local X_batch = X:narrow(1, batch, sz)
        local Y_batch = Y:narrow(1, batch, sz)

        local outputs = model:forward(X_batch)
        local loss = criterion:forward(outputs, Y_batch)

        local _, correct = compute_err(Y_batch, outputs)
        total_correct = total_correct + correct
        total_loss = total_loss + loss * batch_size
    end

    return total_loss / N, total_correct / N
end

function train_model(X, Y, valid_X, valid_Y, word_vecs)
  local eta = opt.eta
  local batch_size = opt.batch_size
  local max_epochs = opt.max_epochs
  local N = X:size(1)

  local model = NNLM()
  local criterion = nn.ClassNLLCriterion()

  -- only call this once
  local params, grads = model:getParameters()
  -- sgd state
  local state = { learningRate = eta }

  local prev_loss = 1e10
  local epoch = 1
  local timer = torch.Timer()
  while epoch <= max_epochs do
      print('Epoch:', epoch)
      local epoch_time = timer:time().real
      local total_loss = 0

      -- shuffle for batches
      local shuffle = torch.randperm(N):long()
      X = X:index(1, shuffle)
      Y = Y:index(1, shuffle)

      -- loop through each batch
      model:training()
      for batch = 1, N, batch_size do
          --if ((batch - 1) / batch_size) % 1000 == 0 then
            --print('Sample:', batch)
            --print('Current train loss:', total_loss / batch)
            --print('Current time:', 1000 * (timer:time().real - epoch_time), 'ms')
          --end
          local sz = batch_size
          if batch + batch_size > N then
            sz = N - batch + 1
          end
          local X_batch = X:narrow(1, batch, sz)
          local Y_batch = Y:narrow(1, batch, sz)

          -- closure to return err, df/dx
          local func = function(x)
            -- get new parameters
            if x ~= params then
              params:copy(x)
            end
            -- reset gradients
            grads:zero()

            -- forward
            local inputs = X_batch
            local outputs = model:forward(inputs)
            local loss = criterion:forward(outputs, Y_batch)
            local _, correct = compute_err(Y_batch, outputs)

            -- track errors
            total_loss = total_loss + loss * batch_size
            total_correct = total_correct + correct

            -- compute gradients
            local df_do = criterion:backward(outputs, Y_batch)
            model:backward(inputs, df_do)

            return loss, grads
          end

          optim.sgd(func, params, state)

          -- normalize weights
          local renorm = function(row)
            local n = row:norm()
            row:mul(opt.L2s):div(1e-7 + n)
          end
          local w = model:get(1).weight
          for j = 1, w:size(1) do
            renorm(w[j])
          end

          -- padding to zero
          model:get(1).weight[1]:zero()
      end

      print('Train loss:', total_loss / N)
      print('Train percent:', total_correct / N)

      local loss, valid_percent = model_eval(model, criterion, valid_X, valid_Y)
      print('Valid loss:', loss)
      print('Valid percent:', valid_percent)

      print('time for one epoch: ', (timer:time().real - epoch_time) * 1000, 'ms')
      print('')
      if loss > prev_loss and epoch > 5 then
        prev_loss = loss
        break
      end
      prev_loss = loss
      epoch = epoch + 1
      torch.save(opt.model_out_name .. '_' .. opt.classifier .. '.t7', { model = model })
  end
  print('Trained', epoch, 'epochs')
  return model, prev_loss
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
   window_size = 5

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
