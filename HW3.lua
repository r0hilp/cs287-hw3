-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

PAD = 3 -- padding index

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-action', 'train', 'train or test')
cmd:option('-smoothing', '', 'smoothing method')
cmd:option('-lm', 'mle', 'classifier to use: mle, laplace, NNLM')
cmd:option('-cm_out_name', '0', 'output file name of count matrix [set to 0 to not save]')
cmd:option('-bigram_cm', '', 'path to precomputed bigram count matrix')
cmd:option('-trigram_cm', '', 'path to precomputed trigram count matrix')

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
cmd:option('-skip_connect', 1, 'use skip connections in NNLM')

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

function hash(context, nclasses)
  -- Hashes ngram context for 
  local total = 0
  for i = 1, context:size(1) do
    total = total + (context[i] - 1) * (nclasses ^ (i-1))
  end
  return total
end

function unhash(X_idx, nclasses, ngram_size)
  -- Converts index to ngram context
  local X = torch.zeros(X_idx:size(1), ngram_size - 1)
  for i = 1, X:size(1) do
      idx = X_idx[i][1]
    for j = 1, ngram_size - 1 do
      X[i][j] = math.mod(idx, nclasses) + 1
      idx = (idx - context[i]) / nclasses
    end
  end
  return X
end

function make_count_matrix(X, Y, nclasses)
  -- Construct count matrix
  local CM = {}
  for i = 1, X:size(1) do
    prefix = hash(X[i], nclasses)
    if CM[prefix] == nil then
      CM[prefix] = {}
      word = Y[i]
      CM[prefix][word] = 1
    else
      word = Y[i]
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
    prefix = hash(X[i], nclasses)
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

function predict_witten_bell(X, bigram_CM, trigram_CM, nclasses, queries)
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
      prefix = hash(X[i], nclasses)
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
      trigram_prefix = hash(X[i], nclasses)
      bigram_prefix = math.mod(trigram_prefix, nclasses^2)
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

function perplexity(preds)
  local perp = torch.zeros(preds:size(1), 1)
  for i = 1, preds:size(1) do
    perp[i] = math.exp(preds[i]:log():mul(-1):mean())
  end
  return perp
end

function NNLM()
  if opt.warm_start ~= '' then
    return torch.load(opt.warm_start).model
  end

  local model = nn.Sequential()
  local word_embed = nn.LookupTable(nfeatures, opt.embed)
  word_embed.weight[1]:zero()
  model:add(word_embed)
  local view = nn.View(opt.embed * window_size)
  model:add(view) -- concat

  local lin1 = nn.Sequential()
  lin1:add(nn.Linear(opt.embed * window_size, opt.hidden))
  lin1:add(nn.Tanh())
  
  if opt.skip_connect == 1 then
    -- skip connections
    local skip = nn.ParallelTable()
    skip:add(lin1)
    skip:add(view)
    model:add(skip)
    model:add(nn.JoinTable(2)) -- 2 for batch
    model:add(nn.Linear(opt.hidden + opt.embed * window_size, nclasses))
  else
    model:add(lin1)
    model:add(nn.Linear(opt.hidden, nclasses))
  end

  --model:add(nn.LogSoftMax())
  -- no softmax here for compatibility with NCE
  return model
end

function compute_err(Y, pred)
  -- Compute error from Y
  local _, argmax = torch.max(pred, 2)
  argmax:squeeze()

  local correct = argmax:eq(Y):sum()
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

function train_model(X, Y, valid_X, valid_Y)
  local eta = opt.eta
  local batch_size = opt.batch_size
  local max_epochs = opt.max_epochs
  local N = X:size(1)

  local model = NNLM()
  model:add(nn.LogSoftMax())
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
   local X_context = f:read('train_context'):all():long()
   local Y = f:read('train_output'):all():long()
   local valid_X = f:read('valid_input'):all():long()
   local valid_Y = f:read('valid_output'):all():long()
   local valid_X_blanks = f:read('valid_blanks_input'):all():long()
   local valid_X_queries = f:read('valid_blanks_queries'):all():long()
   local valid_X_context = f:read('valid_context'):all():long()
   local test_X = f:read('test_blanks_input'):all():long()
   local test_X_queries = f:read('test_blanks_queries'):all():long()
   local test_X_context = f:read('test_blanks_context'):all():long()
   nclasses = f:read('nclasses'):all():long()[1]
   window_size = f:read('context_size'):all():long()[1]


   -- Train.
   if opt.action == 'train' then
     print('Training...')
     if opt.lm == 'mle' then
       print(X:size(1), X:size(2))
       CM = make_count_matrix(X, Y, nclasses)
       preds = predict_laplace(valid_X_blanks, CM, valid_X_queries, 0) 
     elseif opt.lm == 'laplace' then
       alpha = opt.alpha
       CM = make_count_matrix(X, Y, nclasses)
       preds = predict_laplace(valid_X_blanks, CM, valid_X_queries, alpha) 
       print(perplexity(preds):mean())
     elseif opt.lm == 'NNLM' then
       train_model(X_context, Y, valid_X_context, valid_Y)
     end
   end 

   -- Test.
end

main()
