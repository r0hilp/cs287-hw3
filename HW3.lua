-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

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
  -- Get length of table (# does not work for our format)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

function tblsum(T)
  -- Get sum of table values
  local total = 0
  for key, value in pairs(T) do total = total + value end
  return total
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

function predict_laplace(X, CM, queries, vocab_size, alpha, renormalize)
  -- Predict distribution of the word following X[i] over queries[i]
  local preds = torch.zeros(X:size(1), queries:size(2)) 
  preds:fill(alpha)
  for i = 1, X:size(1) do
    local prefix = hash(X[i])
    if CM[prefix] ~= nil then
      total_count = alpha * vocab_size
      for j = 1, queries[i]:size(1) do
        if CM[prefix][queries[i][j]] ~= nil then
          preds[i][j] = preds[i][j] + CM[prefix][queries[i][j]]
        end
      end
      for suffix, count in pairs(CM[prefix]) do
        total_count = total_count + count
      end
      -- Normalize
      preds[i]:div(total_count)

      -- Renormalize
      if renormalize == true then
        queries_sum = preds[i]:sum()
        if queries_sum == 0 then
          preds[i]:fill(0)
        else
          preds[i]:div(queries_sum)
        end
      end
    end
  end

  return preds
end

function predict_witten_bell(X, bigram_CM, trigram_CM, queries, vocab_size, alpha, renormalize)
  -- Predict distribution of the word following X[i] over queries[i]
  local preds = torch.zeros(X:size(1), queries:size(2))

  -- Get unigram predictions
  local unigram_CM = {}
  for i = 1, vocab_size do
    unigram_CM[i] = 0
  end
  for bigram, suffixes in pairs(bigram_CM) do 
    for unigram, count in pairs(suffixes) do
      unigram_CM[unigram] = unigram_CM[unigram] + count
    end
  end
  local total_unigram_count = alpha * vocab_size + tblsum(unigram_CM)
  local unigram_preds = torch.zeros(X:size(1), queries:size(2))
  for i = 1, X:size(1) do
    for j = 1, queries:size(2) do
      unigram_preds[i][j] = (alpha + unigram_CM[queries[i][j]])/total_unigram_count
    end
  end

  local ngram_size = X:size(2) + 1 
  local X_2gram = X:select(2, X:size(2)):resize(X:size(1), 1)

  -- calculate wb probabilities for bigram models
  local bigram_preds = predict_laplace(X_2gram, bigram_CM, queries, vocab_size, alpha, false)
  for i = 1, X:size(1) do
    local prefix = hash(X[i]) % vocab_size
    local lambda = 0
    if bigram_CM[prefix] ~= nil then
      local unique_suffixes = tbllength(bigram_CM[prefix])
      local total_bigram_count = tblsum(bigram_CM[prefix])
      lambda = 1 - unique_suffixes/(unique_suffixes + total_bigram_count)
    end
    preds[i] = bigram_preds[i]:mul(lambda):add(unigram_preds[i]:mul(1-lambda))
  end

  -- one more set of calculations for ngram_size = 3
  if ngram_size == 3 then
    local trigram_preds = predict_laplace(X, trigram_CM, queries, vocab_size, alpha, false)
    for i = 1, X:size(1) do
      local prefix = hash(X[i])
      local lambda = 0
      if trigram_CM[prefix] ~= nil then
        local unique_suffixes = tbllength(trigram_CM[prefix])
        local total_trigram_count = tblsum(trigram_CM[prefix])
        lambda = 1 - unique_suffixes/(unique_suffixes + total_trigram_count)
      end
      trigram_preds:mul(lambda)
      preds[i]:mul(1-lambda)
      preds[i] = trigram_preds[i]:add(preds[i])
      preds[i] = trigram_preds[i]:mul(lambda):add(preds[i]:mul(1-lambda))
    end
  end

  --renormalize preds
  if renormalize == true then
    for i = 1, preds:size(1) do
      preds[i]:div(preds[i]:sum())
    end
  end

  return preds
end

function predict_kneser_ney(X, bigram_CM, trigram_CM, queries, vocab_size, alpha, delta, renormalize)
  -- Predict distribution of the word following X[i] over queries[i]
  local preds = torch.zeros(X:size(1), queries:size(2))

  -- Get p(w_i)
  local unigram_preds = torch.zeros(vocab_size)
  for word = 1, vocab_size do
    for bigram, suffixes in pairs(bigram_CM) do
      if suffixes[word] ~= nil then
        unigram_preds[word] = unigram_preds[word] + 1
      end
    end
  end

  -- Get bigram pred
  for i = 1, X:size(1) do
    prefix = hash(X[i]) % vocab_size^2
    for j = 1, queries:size(2) do
      if bigram_CM[prefix] == nil then
        if renormalize == true then
          preds[i]:fill(1/queries:size(2))
        end
      else
        if bigram_CM[prefix][queries[i][j]] ~= nil then
          preds[i][j] = math.max(bigram_CM[prefix][queries[i][j]] - delta, 0) / tblsum(bigram_CM[prefix])
        end
        preds[i][j] = preds[i][j] + unigram_preds[queries[i][j]]
      end
    end
  end

  -- Get trigram pred
  if X:size(2) == 2 then
    for i = 1, X:size(1) do
      prefix = hash(X[i])
      for j = 1, queries:size(2) do
        if trigram_CM[prefix] == nil then
          if renormalize == true then
            
          end
        end
      end
    end
  end

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

function NCE_LM(unigram_p)
  -- unigram_p is log probs
  -- input: {{context, output word}, output_word, output_word}
  if opt.warm_start ~= '' then
    return torch.load(opt.warm_start).model
  end

  local model = nn.Sequential()
  local parallel = nn.ParallelTable()

  local to_dot = nn.ParallelTable()
  local dot = nn.Sequential()

  -- compute hidden layer from context, and dot with output embed
  local hid = nn.Sequential()
  hid:add(nn.LookupTable(vocab_size, opt.embed))
  hid:add(nn.View(opt.embed * window_size)) -- concat
  hid:add(nn.Linear(opt.embed * window_size, opt.hidden))
  hid:add(nn.Tanh())
  to_dot:add(hid)
  to_dot:add(nn.LookupTable(vocab_size, opt.hidden))
  dot:add(to_dot)
  dot:add(nn.CMulTable())
  dot:add(nn.Sum(2)) -- 2 for batch
  parallel:add(dot)

  -- bias embedding
  local bias = nn.Sequential()
  bias:add(nn.LookupTable(vocab_size, 1))
  bias:add(nn.Squeeze())
  parallel:add(bias)

  -- noise will compute K * p_ML(w)
  local noise = nn.Sequential()
  local lookup_ml = nn.LookupTable(vocab_size, 1)
  lookup_ml.weight = unigram_p:clone()
  lookup_ml.weight:add(torch.log(opt.K_noise))
  noise:add(lookup_ml)
  noise:add(nn.MulConstant(-1))
  parallel:add(noise)

  -- combine score and noise terms
  model:add(parallel)
  model:add(nn.CAddTable())
  model:add(nn.Sigmoid())

  return model
end

-- not sure how to use Q yet!!!
function compute_err(Y, pred, X_Q)
  -- Compute error from Y
  local _, argmax = torch.max(pred, 2)
  argmax:squeeze()

  local correct = argmax:eq(Y:long()):sum()
  return argmax, correct
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

function model_eval_NCE(model, X, Y, X_Q, Y_index)
    -- batch eval
    local N = X:size(1)
    local batch_size = opt.batch_size

    -- get output word embeddings
    local embeds = model:get(1):get(1):get(1):get(2)
    local bias = model:get(1):get(2):get(1)
    local W = embeds.weight:clone()
    local b = bias.weight:clone()
    
    -- build full model
    local hid_model = model:get(1):get(1):get(1):get(1)
    local lin = nn.Linear(opt.hidden, vocab_size)
    lin.weight = W
    lin.bias = b:squeeze()

    local real_model = nn.Sequential()
    real_model:add(hid_model)
    real_model:add(lin)
    real_model:evaluate()

    local criterion = nn.ClassNLLCriterion()

    local total_loss = 0
    for batch = 1, X:size(1), batch_size do
        local sz = batch_size
        if batch + batch_size > N then
          sz = N - batch + 1
        end
        local X_batch = X:narrow(1, batch, sz)

        local scores = real_model:forward(X_batch)
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
            local renorm = function(row)
              local n = row:norm()
              row:mul(opt.L2s):div(1e-7 + n)
            end
            local w = model:get(1).weight
            for j = 1, w:size(1) do
              renorm(w[j])
            end
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

-- NCE training
function train_model_NCE(X, Y, valid_X, valid_Y, valid_blanks_X, valid_blanks_Q, valid_blanks_Y, valid_blanks_index, unigram_p)
  local eta = opt.eta
  local batch_size = opt.batch_size
  local max_epochs = opt.max_epochs
  local N = X:size(1)

  local model = NCE_LM(unigram_p) -- pass log unigram p
  local criterion = nn.BCECriterion()

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
            end
          end
          local sz = batch_size
          if batch + batch_size > N then
            sz = N - batch + 1
          end
          local X_batch = X:narrow(1, batch, sz):double()
          local Y_batch = Y:narrow(1, batch, sz):double()

          -- closure to return err, df/dx
          local func = function(x)
              -- get new parameters
              if x ~= params then
                params:copy(x)
              end
              -- reset gradients
              grads:zero()

              -- I need to: sample K_noise unigram guys. Append the words to Y_batch. Add K_noise 0's (per guy in the batch) as the gold for criterion
              -- D_noise will indicate if noise or not.
              local K = opt.K_noise
              local D_noise = torch.cat(torch.ones(sz), torch.zeros(K * sz))
              local X_batch = torch.repeatTensor(X_batch, K+1, 1)
              local samples = torch.multinomial(torch.exp(unigram_p), K*sz, true):double()
              local Y_batch = torch.cat(Y_batch, samples)

              -- forward
              local inputs = {{X_batch, Y_batch}, Y_batch, Y_batch}
              local NCE_val = model:forward(inputs)
              local loss = criterion:forward(NCE_val, D_noise)

              -- track errors
              total_loss = total_loss + loss * batch_size

              -- compute gradients
              local df_do = criterion:backward(NCE_val, D_noise)
              model:backward(inputs, df_do)

              -- zero out unigram ML grads
              model:get(1):get(3):get(1).gradWeight:zero()

              return loss, grads
          end

          optim.sgd(func, params, state)

          -- normalize weights
          if opt.L2s > 0 then
            local renorm = function(row)
              local n = row:norm()
              row:mul(opt.L2s):div(1e-7 + n)
            end
            local w = model:get(1):get(1):get(1):get(1):get(1).weight
            for j = 1, w:size(1) do
              renorm(w[j])
            end
          end
      end

      print('Train loss:', total_loss / N)

      if opt.has_blanks == 1 then
        local blanks_loss = model_eval_NCE(model, valid_blanks_X, valid_blanks_Y, valid_blanks_Q, valid_blanks_index)
        print('Valid blanks perplexity:', torch.exp(blanks_loss))
      end
      local loss = model_eval_NCE(model, valid_X, valid_Y)
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
   local test_X, test_X_queries, test_X_context
   if opt.has_blanks == 1 then
     valid_blanks_X  = f:read('valid_blanks_input'):all():long()
     valid_blanks_X_context = f:read('valid_blanks_context'):all():long()
     valid_blanks_Q = f:read('valid_blanks_queries'):all():long()
     valid_blanks_Y = f:read('valid_blanks_index'):all():long()
     valid_blanks_index = f:read('valid_blanks_index'):all():long()

     test_X = f:read('test_blanks_input'):all():long()
     test_X_queries = f:read('test_blanks_queries'):all():long()
     test_X_context = f:read('test_blanks_context'):all():long()
   end
   vocab_size = f:read('vocab_size'):all():long()[1]
   window_size = f:read('context_size'):all():long()[1]

   print('Train context size:', X_context:size(1))

   -- Train.
   if opt.action == 'train' then
     print('Training...')
     if opt.lm == 'mle' then
       CM = make_count_matrix(X, Y)
       preds = predict_laplace(valid_blanks_X, CM, valid_blanks_Q, vocab_size, 0, false) 
       print(perplexity(preds, valid_blanks_index))
     elseif opt.lm == 'laplace' then
       alpha = opt.alpha
       CM = make_count_matrix(X, Y)
       -- run predictions on valid.txt
       valid_preds = predict_laplace(valid_X, CM, valid_Y:resize(valid_Y:size(1), 1), vocab_size, alpha, false)
       print(torch.exp(valid_preds:log():mul(-1):mean()))
       -- ...and valid_blanks.txt
       valid_blanks_preds = predict_laplace(valid_blanks_X, CM, valid_blanks_Q, vocab_size, alpha, true) 
       print(perplexity(valid_blanks_preds, valid_blanks_index))
     elseif opt.lm == 'wb' then
       alpha = opt.alpha
       bigram_CM = torch.load(opt.bigram_cm)
       trigram_CM = torch.load(opt.trigram_cm)
       -- run predictions on valid.txt
       valid_preds = predict_witten_bell(valid_X, bigram_CM, trigram_CM, valid_Y:resize(valid_Y:size(1), 1), vocab_size, alpha, false)
       print(valid_preds:narrow(1,1,10))
       print(valid_Y:resize(valid_Y:size(1), 1):narrow(1,1,10))
       print(torch.exp(valid_preds:log():mul(-1):mean()))
       -- ...and valid_blanks.txt
       -- valid_blanks_preds = predict_witten_bell(valid_blanks_X, bigram_CM, trigram_CM, valid_blanks_Q, vocab_size, alpha, true)
       -- print(perplexity(valid_blanks_preds, valid_blanks_index))
     elseif opt.lm == 'NNLM' then
       train_model(X_context, Y, valid_X_context, valid_Y, valid_blanks_X_context, valid_blanks_Q, valid_blanks_Y, valid_blanks_index)
     elseif opt.lm == 'NCE' then
       local unigram_p = torch.Tensor(vocab_size):fill(opt.alpha)
       for i = 1, Y:size(1) do
         unigram_p[Y[i]] = unigram_p[Y[i]] + 1
       end
       unigram_p:log()

       train_model_NCE(X_context, Y, valid_X_context, valid_Y, valid_blanks_X_context, valid_blanks_Q, valid_blanks_Y, valid_blanks_index, unigram_p)
     end
   end 

   -- Test.
   if opt.action == 'test' then
     print('Testing...')
     test_model = torch.load(opt.test_model).model
     local logsoftmax = nn.LogSoftMax()
     local scores = logsoftmax:forward(test_model:forward(test_X_context))
     scores = scores:exp()
     scores = scores:double()
     local preds = torch.Tensor(test_X_context:size(1), test_X_queries:size(2))
     f = io.open('PTB_pred.test', 'w')
     local out = {"ID"}
     for i = 1, test_X_queries:size(2) do
        table.insert(out, "Class"..i)
     end
     f:write(table.concat(out, ","))
     f:write("\n")
     for i = 1, test_X_context:size(1) do
        out = {i}
        preds[i] = scores[i]:index(1, test_X_queries[i])
        -- renormalize
        --sum = preds[i]:sum()
        --if sum == 0 then
          --preds[i]:fill(1/test_X_queries:size(2))
        --else
          --preds[i]:div(sum)
        --end
        for j = 1, test_X_queries:size(2) do
          table.insert(out, preds[i][j])
        end
        f:write(table.concat(out, ","))
        f:write("\n")
     end
   end
end

main()
