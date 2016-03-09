-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

require("cutorch")
require("cunn")
require("cudnn")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-action', 'train', 'train or test')
cmd:option('-smoothing', '', 'smoothing method')
cmd:option('-lm', 'mle', 'classifier to use: mle, laplace, NNLM, NCE')
cmd:option('-cm_out_name', '0', 'output file name of count matrix [set to 0 to not save]')
cmd:option('-bigram_cm', '', 'path to precomputed bigram count matrix')
cmd:option('-trigram_cm', '', 'path to precomputed trigram count matrix')

cmd:option('-window_size', 5, 'window size')
cmd:option('-warm_start', '', 'torch file with previous model')
cmd:option('-test_model', '', 'model to test on')
cmd:option('-model_out_name', 'train', 'output file name of model')
cmd:option('-debug', 0, 'print training debug')
cmd:option('-has_blanks', 1, 'use blanks data for valid')

-- Hyperparameters
cmd:option('-alpha', 0.1, 'laplace smoothing alpha')

cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-batch_size', 32, 'batch size for SGD')
cmd:option('-max_epochs', 20, 'max # of epochs for SGD')
cmd:option('-L2s', 1, 'normalize L2 of word embeddings')

cmd:option('-embed', 50, 'size of word embeddings')
cmd:option('-hidden', 100, 'size of hidden layer for neural network')
cmd:option('-skip_connect', 0, 'use skip connections in NNLM')

cmd:option('-K_noise', 10, 'number of noise samples')

function tbllength(T)
  -- Get length of table
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

function tblsum(T)
  -- Get sum of table values
  local total = 0
  for key, value in pairs(T) do total = total + value end
  return value
end

function hash(context)
  -- Hashes ngram context for 
  local total = 0
  for i = 1, context:size(1) do
    total = total + (context[i] - 1) * (vocab_size ^ (i-1))
  end
  return total
end

function unhash(X_idx)
  -- Converts index to ngram context
  local X = torch.zeros(X_idx:size(1), ngram_size - 1)
  for i = 1, X:size(1) do
      local idx = X_idx[i][1]
    for j = 1, ngram_size - 1 do
      X[i][j] = math.mod(idx, vocab_size) + 1
      local idx = (idx - context[i]) / vocab_size
    end
  end
  return X
end

function make_count_matrix(X, Y)
  -- Construct count matrix
  local CM = {}
  for i = 1, X:size(1) do
    local prefix = hash(X[i])
    if CM[prefix] == nil then
      CM[prefix] = {}
      local word = Y[i]
      CM[prefix][word] = 1
    else
      local word = Y[i]
      if CM[prefix][word] == nil then
        CM[prefix][word] = 1
      else
        CM[prefix][word] = CM[prefix][word] + 1
      end
    end
  end
  if opt.cm_out_name ~= '0' then
    local output = opt.cm_out_name .. '.t7'
    torch.save(output, CM)
  end

  return CM
end

function predict_laplace(X, CM, queries, alpha)
  -- Predict distribution of the word following X[i] over queries[i]
  local preds = torch.zeros(X:size(1), queries:size(2)) 
  for i = 1, X:size(1) do
    local prefix = hash(X[i])
    -- Return uniform distribution if nil
    if CM[prefix] == nil then
      preds[i]:fill(1/queries:size(2))
    else
      for j = 1, queries[i]:size(1) do
        preds[i][j] = alpha
        if CM[prefix][queries[i][j]] ~= nil then
          preds[i][j] = preds[i][j] + CM[prefix][queries[i][j]]
        end
      end
      sum = preds[i]:sum()
      if sum == 0 then
        preds[i]:fill(1/queries:size(2))
      else
        preds[i]:div(sum)
      end
    end
  end

  return preds
end

function predict_witten_bell(X, bigram_CM, trigram_CM, queries)
  -- Predict distribution of the word following X[i] over queries[i]
  local preds = torch.zeroes(X:size(1), queries:size(2))
  local ngram_size = X:size(2) + 1 
  if ngram_size == 2 then 
    -- Calculate unigram counts
    unigram_CM = {}
    for bigram, suffixes in pairs(bigram_CM) do 
      for unigram, count in pairs(suffixes) do
        if unigram_CM[unigram] == nil then
          unigram_CM[unigram] = count
        else
          unigram_CM[unigram] = unigram_CM[unigram] + count
        end
      end
    end
    total_unigram_count = tblsum(unigram_CM)
    for i = 1, X:size(1) do
      prefix = hash(X[i])
      if bigram_CM[prefix] == nil then
        preds[i]:fill(1/queries:size(2))
      else
        unique_types = tbllength(bigram_CM[prefix])
        total_bigram_count = tblsum(bigram_CM[prefix])
        lambda = 1 - unique_types/(unique_types + total_count)
        for j = 1, queries:size(2) do
          if bigram_CM[prefix][queries[i][j]] ~= nil then
            bigram_count = bigram_CM[prefix][queries[i][j]]
          else
            bigram_count = 0
          end
          if unigram_CM[queries[i][j]] ~= nil then
            unigram_count = unigram_CM[queries[i][j]] or 0
          else
            unigram_count = 0
          end
          preds[i][j] = (bigram_count/total_bigram_count) * lambda + (unigram_count/total_unigram_count) * (1 - lambda)
        end
        sum = preds[i]:sum()
        if sum == 0 then
          preds[i]:fill(1/queries:size(2))
        else
          preds[i]:div(sum)
        end
      end
    end
  elseif ngram_size == 3 then
    for i = 1, X:size(1) do
      trigram_prefix = hash(X[i])
      bigram_prefix = math.mod(trigram_prefix, vocab_size^2)
      if trigram_CM[trigram_prefix] == nil then
        preds[i]:fill(1/queries:size(2))
      else
        unique_types = tbllength(trigram_CM[trigram_prefix])
        total_trigram_count = tblsum(trigram_CM[trigram_prefix])
        if bigram_CM[bigram_prefix] ~= nil then
          total_bigram_count = tblsum(bigram_CM[bigram_prefix])
        else
          total_bigram_count = 1
        end
        lambda = 1 - unique_types/(unique_types + total_trigram_count)
        for j = 1, queries:size(2) do
          if trigram_CM[trigram_prefix][queries[i][j]] ~= nil then
            trigram_count = trigram_CM[trigram_prefix][queries[i][j]]
          else
            trigram_count = 0
          end
          if bigram_CM[bigram_prefix][queries[i][j]] ~= nil then
            bigram_count = bigram_CM[bigram_prefix][queries[i][j]]
          else
            bigram_count = 0
          end
          preds[i][j] = (trigram_count/total_trigram_count) * lambda + (bigram_count/total_bigram_count) * (1 - lambda)
        end
        sum = preds[i]:sum()
        if sum == 0 then
          preds[i]:fill(1/queries:size(2))
        else
          preds[i]:div(sum)
        end
      end
    end
  end

  return preds
end

function perplexity(preds, Y)
  local nll = nn.ClassNLLCriterion():forward(preds:log(), Y)
  return torch.exp(nll)
end

function NNLM()
  if opt.warm_start ~= '' then
    return torch.load(opt.warm_start).model
  end

  local model = nn.Sequential()
  model:add(nn.LookupTable(vocab_size, opt.embed))
  model:add(nn.View(opt.embed * window_size)) -- concat

  local lin1 = nn.Sequential()
  lin1:add(nn.Linear(opt.embed * window_size, opt.hidden))
  lin1:add(nn.Tanh())

  if opt.skip_connect == 1 then
    -- skip connections
    local skip = nn.ConcatTable()
    skip:add(lin1)
    skip:add(nn.Identity())
    model:add(skip)
    model:add(nn.JoinTable(2)) -- 2 for batch
    model:add(nn.Linear(opt.hidden + opt.embed * window_size, vocab_size))
  else
    model:add(lin1)
    model:add(nn.Linear(opt.hidden, vocab_size))
    -- no softmax, scores only
  end

  return model
end

function NCE_hid()
  local model = nn.Sequential()
  
  model:add(nn.LookupTable(vocab_size, opt.embed))
  model:add(nn.View(opt.embed * window_size)) -- concat
  model:add(nn.Linear(opt.embed * window_size, opt.hidden))
  model:add(cudnn.Tanh())

  return model
end

function model_eval(model, criterion, X, Y, X_Q, Y_index)
    -- batch eval
    model:evaluate()
    local N = X:size(1)
    local batch_size = opt.batch_size

    local total_loss = 0
    for batch = 1, X:size(1), batch_size do
        local sz = batch_size
        if batch + batch_size > N then
          sz = N - batch + 1
        end
        local X_batch = X:narrow(1, batch, sz)

        local Y_batch
        local scores = model:forward(X_batch)
        local outputs
        if X_Q then
          local X_Q_batch = X_Q:narrow(1, batch, sz)
          outputs = torch.Tensor(sz, X_Q:size(2))
          for i = 1, sz do
            outputs[i] = nn.LogSoftMax():forward(scores[i]:index(1, X_Q_batch[i]))
          end
          Y_batch = Y_index:narrow(1, batch, sz)
        else
          outputs = nn.LogSoftMax():forward(scores)
          Y_batch = Y:narrow(1, batch, sz)
        end

        local loss = criterion:forward(outputs, Y_batch)
        total_loss = total_loss + loss * batch_size
    end

    return total_loss / N
end


function train_model(X, Y, valid_X, valid_Y, valid_blanks_X, valid_blanks_Q, valid_blanks_Y, valid_blanks_index)
  local eta = opt.eta
  local batch_size = opt.batch_size
  local max_epochs = opt.max_epochs
  local N = X:size(1)

  local model = NNLM()
  local logsoftmax = nn.LogSoftMax()
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
      local total_correct = 0

      -- shuffle for batches
      local shuffle = torch.randperm(N):long()
      X = X:index(1, shuffle)
      Y = Y:index(1, shuffle)

      -- loop through each batch
      model:training()
      for batch = 1, N, batch_size do
          if opt.debug == 1 then
            if ((batch - 1) / batch_size) % 300 == 0 then
              print('Sample:', batch)
              print('Current train loss:', total_loss / batch)
              print('Current time:', 1000 * (timer:time().real - epoch_time), 'ms')
              local blanks_loss = model_eval(model, criterion, valid_blanks_X, valid_blanks_Y, valid_blanks_Q, valid_blanks_index)
              print('Valid blanks perplexity:', torch.exp(blanks_loss))
            end
          end
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
            local scores = model:forward(inputs)
            local outputs = logsoftmax:forward(scores)
            local loss = criterion:forward(outputs, Y_batch)

            -- track errors
            total_loss = total_loss + loss * batch_size

            -- compute gradients
            local df_do = criterion:backward(outputs, Y_batch)
            local df_dz = logsoftmax:backward(scores, df_do)
            model:backward(inputs, df_dz)

            return loss, grads
          end

          optim.sgd(func, params, state)

          -- normalize weights
          if opt.L2s > 0 then
            local w = model:get(1).weight
            w:renorm(2, 2, opt.L2s)
          end
      end

      print('Train perplexity:', torch.exp(total_loss / N))

      if opt.has_blanks == 1 then
        local blanks_loss = model_eval(model, criterion, valid_blanks_X, valid_blanks_Y, valid_blanks_Q, valid_blanks_index)
        print('Valid blanks perplexity:', torch.exp(blanks_loss))
      end
      local loss = model_eval(model, criterion, valid_X, valid_Y)
      print('Valid perplexity:', torch.exp(loss))

      print('time for one epoch: ', (timer:time().real - epoch_time) * 1000, 'ms')
      print('')
      if loss > prev_loss and epoch > 5 then
        prev_loss = loss
        break
      end
      prev_loss = loss
      epoch = epoch + 1
      torch.save(opt.model_out_name .. '_' .. opt.lm .. '.t7', { model = model })
  end
  print('Trained', epoch, 'epochs')
  return model, prev_loss
end

function model_eval_NCE(hid_model, W, b, X, Y, X_Q, Y_index)
    -- batch eval
    local N = X:size(1)
    local batch_size = opt.batch_size
    
    -- build full model
    local lin = nn.Linear(opt.hidden, vocab_size)
    lin.weight = W
    lin.bias = b:squeeze()

    local real_model = nn.Sequential()
    real_model:add(hid_model)
    real_model:add(lin)
    real_model:cuda()

    local criterion = nn.ClassNLLCriterion()
    criterion:cuda()

    local logsoftmax = cudnn.LogSoftMax()
    logsoftmax:cuda()

    local total_loss = 0
    for batch = 1, X:size(1), batch_size do
        local sz = batch_size
        if batch + batch_size > N then
          sz = N - batch + 1
        end
        local X_batch = X:narrow(1, batch, sz):cuda()

        local Y_batch
        local scores = real_model:forward(X_batch)
        local outputs
        if X_Q then
          local X_Q_batch = X_Q:narrow(1, batch, sz)
          outputs = torch.CudaTensor(sz, X_Q:size(2))
          for i = 1, sz do
            outputs[i] = logsoftmax:forward(scores[i]:index(1, X_Q_batch[i]))
          end
          Y_batch = Y_index:narrow(1, batch, sz)
        else
          outputs = logsoftmax:forward(scores)
          Y_batch = Y:narrow(1, batch, sz)
        end

        local loss = criterion:forward(outputs, Y_batch:cuda())
        total_loss = total_loss + loss * batch_size
    end

    return total_loss / N
end

-- NCE training
function train_model_NCE(X, Y, valid_X, valid_Y, valid_blanks_X, valid_blanks_Q, valid_blanks_Y, valid_blanks_index, unigram_p)
  local eta = opt.eta
  local batch_size = opt.batch_size
  local max_epochs = opt.max_epochs
  local K = opt.K_noise
  local N = X:size(1)

  -- init model
  local model = NCE_hid()
  -- save input embeds
  local in_embeds = model:get(1)

  --asdf
  local asdf = nn.Linear(opt.hidden, vocab_size)

  -- other nodes
  local out_lookup = nn.LookupTable(vocab_size, opt.hidden)
  out_lookup.weight = asdf.weight:clone()
  local out_bias = nn.LookupTable(vocab_size, 1)
  out_bias.weight = asdf.bias:clone()
  local dot = nn.Sequential():add(nn.CMulTable()):add(nn.Sum(2))
  local add = nn.CAddTable()
  local p_ml = nn.LookupTable(vocab_size, 1)
  p_ml.weight = unigram_p:clone()
  p_ml.weight:add(torch.log(K))
  local sig = nn.Sequential():add(nn.CSubTable()):add(cudnn.Sigmoid())
  local criterion = nn.BCECriterion()

  local bullshit = nn.Sequential():add(model):add(nn.Linear(opt.hidden, vocab_size)):add(cudnn.LogSoftMax()):cuda()

  model:cuda()
  out_lookup:cuda()
  out_bias:cuda()
  dot:cuda()
  add:cuda()
  p_ml:cuda()
  sig:cuda()
  criterion:cuda()

  local prev_loss = 1e10
  local epoch = 1
  local timer = torch.Timer()
  while epoch <= max_epochs do
      print('Epoch:', epoch)
      local epoch_time = timer:time().real
      local total_loss = 0
      local total_correct = 0

      -- shuffle for batches
      local shuffle = torch.randperm(N):long()
      X = X:index(1, shuffle)
      Y = Y:index(1, shuffle)

      -- loop through each batch
      for batch = 1, N, batch_size do
          if opt.debug == 1 then
            if ((batch - 1) / batch_size) % 300 == 0 then
              print('Sample:', batch)
              print('Current train loss:', total_loss / batch)
              print('Current time:', 1000 * (timer:time().real - epoch_time), 'ms')

              local blanks_loss = model_eval_NCE(model, out_lookup.weight, out_bias.weight, valid_blanks_X, valid_blanks_Y, valid_blanks_Q, valid_blanks_index)
              print('Valid blanks perplexity:', torch.exp(blanks_loss))
            end
          end
          local sz = batch_size
          if batch + batch_size > N then
            sz = N - batch + 1
            --samples = samples:narrow(1, 1, K*sz)
          end
          local X_batch = X:narrow(1, batch, sz):cuda()
          local Y_batch = Y:narrow(1, batch, sz)

          -- sample for NCE
          local samples = torch.multinomial(torch.exp(unigram_p), K*sz, true):long()

          -- I need to: sample K_noise unigram guys. Append the words to Y_batch. Add K_noise 0's (per guy in the batch) as the gold for criterion
          -- D_noise will indicate if noise or not.
          local D_noise = torch.cat(torch.ones(sz), torch.zeros(K*sz)):cuda()
          local x_in = X_batch:repeatTensor(K+1, 1)
          --local x_in = X_batch
          local y_in = torch.cat(Y_batch, samples):cuda()

          -- zero grads
          model:zeroGradParameters()
          out_lookup:zeroGradParameters()
          out_bias:zeroGradParameters()

          -- forward
          local hid = model:forward(x_in)
          local e_out = out_lookup:forward(y_in)
          local b_out = out_bias:forward(y_in)
          local dot_prod = dot:forward({hid, e_out})
          local z = add:forward({dot_prod, b_out})

          --DEBUG 1
          --print('from here:')
          --print(z:narrow(1, 1, 32))
          --local linmmm = nn.Linear(opt.hidden, vocab_size)
          --linmmm.weight = out_lookup.weight
          --linmmm.bias = out_bias.weight
          --local testmmm = nn.Sequential():add(model):add(linmmm):cuda()
          --print('real:')
          --local resmmm = testmmm:forward(x_in):narrow(1, 1, 32)
          --for j = 1, Y_batch:size(1) do
            --print(resmmm[j][Y_batch[j]])
          --end
          --io.read()

          local p = p_ml:forward(y_in)
          local output = sig:forward({z, p})
          local loss = criterion:forward(output, D_noise)

          -- track errors
          total_loss = total_loss + loss * sz

          -- backward
          local dL_do = criterion:backward(output, D_noise)
          local dL_dz = sig:backward({z, p}, dL_do)[1]
          local dL_dadd = add:backward({dot_prod, b_out}, dL_dz)
          out_bias:backward(y_in, dL_dadd[2])
          out_bias:updateParameters(eta)
          local dL_ddot = dot:backward({hid, e_out}, dL_dadd[1])
          out_lookup:backward(y_in, dL_ddot[2])
          out_lookup:updateParameters(eta)

          model:backward(x_in, dL_ddot[1])
          --local g = dL_ddot[1]:clone()
          --g = g:reshape(K+1, sz, opt.hidden)
          --g = g:sum(1):squeeze()
          --model:backward(x_in, g)
          model:updateParameters(eta)

          -- normalize weights
          if opt.L2s > 0 then
            in_embeds.weight:renorm(2, 2, opt.L2s)
          end
      end

      print('Train loss:', total_loss / N)

      if opt.has_blanks == 1 then
        local blanks_loss = model_eval_NCE(model, out_lookup.weight, out_bias.weight, valid_blanks_X, valid_blanks_Y, valid_blanks_Q, valid_blanks_index)
        print('Valid blanks perplexity:', torch.exp(blanks_loss))
      end
      local loss = model_eval_NCE(model, out_lookup.weight, out_bias.weight, valid_X, valid_Y)
      print('Valid perplexity:', torch.exp(loss))

      print('time for one epoch: ', (timer:time().real - epoch_time) * 1000, 'ms')
      print('')
      if loss > prev_loss and epoch > 5 then
        prev_loss = loss
        break
      end
      prev_loss = loss
      epoch = epoch + 1
      torch.save(opt.model_out_name .. '_' .. opt.lm .. '.t7', { model = model, out_bias = out_bias, out_lookup = out_lookup })
  end
  print('Trained', epoch, 'epochs')
  return model, prev_loss
end

function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   local X = f:read('train_input'):all():long()
   local X_context = f:read('train_context'):all():long()
   local Y = f:read('train_output'):all():long()
   local valid_X = f:read('valid_input'):all():long()
   local valid_Y = f:read('valid_output'):all():long()
   local valid_X_context = f:read('valid_context'):all():long()

   local valid_blanks_X, valid_blanks_X_context, valid_blanks_Q, valid_blanks_Y, valid_blanks_index
   if opt.has_blanks == 1 then
     valid_blanks_X  = f:read('valid_blanks_input'):all():long()
     valid_blanks_X_context = f:read('valid_blanks_context'):all():long()
     valid_blanks_Q = f:read('valid_blanks_queries'):all():long()
     valid_blanks_Y = f:read('valid_blanks_index'):all():long()
     valid_blanks_index = f:read('valid_blanks_index'):all():long()

     --local test_X = f:read('test_blanks_input'):all():long()
     --local test_X_queries = f:read('test_blanks_queries'):all():long()
     --local test_X_context = f:read('test_blanks_context'):all():long()
   end
   vocab_size = f:read('vocab_size'):all():long()[1]
   window_size = f:read('context_size'):all():long()[1]

   print('Train context size:', X_context:size(1))

   torch.manualSeed(3435)

   -- Train.
   if opt.action == 'train' then
     print('Training...')
     if opt.lm == 'mle' then
       print(X:size(1), X:size(2))
       CM = make_count_matrix(X, Y)
       preds = predict_laplace(valid_blanks_X, CM, valid_blanks_Q, 0) 
     elseif opt.lm == 'laplace' then
       alpha = opt.alpha
       CM = make_count_matrix(X, Y)
       preds = predict_laplace(valid_blanks_X, CM, valid_blanks_Q, alpha) 
       print(perplexity(preds, valid_blanks_index))
     elseif opt.lm == 'NNLM' then
       train_model(X_context, Y, valid_X_context, valid_Y, valid_blanks_X_context, valid_blanks_Q, valid_blanks_Y, valid_blanks_index)
     elseif opt.lm == 'NCE' then
       local unigram_p = torch.zeros(vocab_size)
       for i = 1, Y:size(1) do
         unigram_p[Y[i]] = unigram_p[Y[i]] + 1
       end
       unigram_p:div(unigram_p:sum())
       unigram_p:log()

       train_model_NCE(X_context, Y, valid_X_context, valid_Y, valid_blanks_X_context, valid_blanks_Q, valid_blanks_Y, valid_blanks_index, unigram_p)
     end
   end 

   -- Test.
end

main()
