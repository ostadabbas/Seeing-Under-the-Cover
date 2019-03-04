require 'image'

local M = {}

function M.Compose(transforms)
   return function(input)
      for _, transform in ipairs(transforms) do
         input = transform(input)
      end
      return input
   end
end

-- multiplies the image by a fixed ammount
function M.Mul(pixel_scale)
   return function(input)
     return input:mul(pixel_scale)
   end
end

-- jitters the image coordinates
function M.Jitter(jitW, jitH)
   return function(input)
     local out = input:clone():fill(0)
     local iW, iH = input:size(3), input:size(2)
     local jW = torch.random(0,jitW) * (torch.uniform() - 0.5)
     local jH = torch.random(0,jitH) * (torch.uniform() - 0.5)
     out[{{},{math.max(1, -jH), math.min(iH, iH-jH)}, {math.max(1, -jW), math.min(iW, iW-jW)}}]:copy(
        input[{{},{math.max(1,jH), math.min(iH-jH, iH)},{math.max(1,jW), math.min(iW-jW, iW)}}])
     return out
   end
end

-- vignette transformation  center highlight, but surroundings dark
function M.Vignette(scale)
   return function (input)
      local iW, iH = input:size(3), input:size(2)
      local nchannels = input:size(1)
      local out = input:clone()
      local imgCntX = iW/2;
      local imgCntY = iH/2;
      local maxDistance = math.sqrt(math.pow(imgCntY,2) + math.pow(imgCntX,2))
      
      local rows_dist = torch.linspace(1, iH, iH):float():add(-imgCntY):pow(2):reshape(iH,1)
      local cols_dist = torch.linspace(1, iW, iW):float():add(-imgCntX):pow(2):reshape(1, iW)
      local dis = torch.add(torch.expand(rows_dist,iW,iH), torch.expand(cols_dist,iW,iH))
      dis = -dis:sqrt():div(maxDistance):mul(1-scale) + 1

      return out:cmul(torch.repeatTensor(dis,nchannels,1,1))
   end
end

-- Gaussian Smoothing
-- Parameters:
-- @param kernel (table): kernel size {W = width,H = height} with integer values
-- @param sigma (table or number): sigma {W, H} keys with float values.
function M.Smoothing(kernel, sigma)
   if sigma then
     if type(sigma) == 'number' then sigma = {W = sigma, H = sigma} end
   else 
     sigma = {W=0.25, H=0.25}
   end
   
   return function(input)
      if kernel.W == 0 and kernel.H == 0 then
        return input
      end

      -- gaussian filter
      local gs = image.gaussian{amplitude=1, 
                                normalize=true, 
                                width=kernel.W, 
                                height=kernel.H, 
                                sigma_horz=sigma.W, 
                                sigma_vert=sigma.H
                              }
      
      return  image.convolve(input, gs, 'same')
   end
end


-- HSV Augmentation
-- Parameters:
-- @param im (tensor): input image
-- @param augHSV (table): standard deviations under {H,S,V} keys with float values.
-- ref: https://github.com/brainstorm-ai/DIGITS/blob/6a150cfbed2aa7dd70992036dfbdf66ee088fba0/tools/torch/data.lua#L48
function M.augmentHSV(augHSV)
  return function(input)
     -- Fair augHSV standard deviation values are {H=0.02,S=0.04,V=0.08}
     local im_hsv = image.rgb2hsv(input)
     if augHSV.H >0 then
        -- We do not need to account for overflow because every number wraps around (1.1=2.1=3.1,etc)
        -- We add a round value (+ 1) to prevent an underflow bug (<0 becomes glitchy)
        im_hsv[1] = im_hsv[1]+(1 + torch.normal(0, augHSV.S))
     end
     if augHSV.S >0 then
        im_hsv[2] = im_hsv[2]+torch.normal(0, augHSV.S) 
        im_hsv[2].image.saturate(im_hsv[2]) -- bound saturation between 0 and 1
     end
     if augHSV.V >0 then
        im_hsv[3] = im_hsv[3]+torch.normal(0, augHSV.V) 
        im_hsv[3].image.saturate(im_hsv[3]) -- bound value between 0 and 1
     end
     return image.hsv2rgb(im_hsv)
   end
 end
 
-- Adds noise to the image
-- Parameters:
-- @param im (tensor): input image
-- @param augNoise (float): the standard deviation of the white noise
-- ref: https://github.com/brainstorm-ai/DIGITS/blob/6a150cfbed2aa7dd70992036dfbdf66ee088fba0/tools/torch/data.lua#L135
function M.AddNoise(augNoise)
  return function(input)
     -- AWGN:
     -- torch.randn makes noise with mean 0 and variance 1 (=stddev 1)
     --  so we multiply the tensor with our augNoise factor, that has a linear relation with
     --  the standard deviation (but the variance will be increased quadratically).
     return torch.add(input, torch.randn(input:size()):float()*augNoise)
  end
end

-- Scale and Rotation augmentation (warping)
-- Parameters:
-- @param im (tensor): input image
-- @param augRot (float): extremes of random rotation, uniformly distributed between (degrees)
-- @param augScale (float): the standard deviation of the extra scaling factor
-- ref: https://github.com/brainstorm-ai/DIGITS/blob/6a150cfbed2aa7dd70992036dfbdf66ee088fba0/tools/torch/data.lua#L67
function M.Warp(augRot, augScale)
   return function(input)
    -- A nice function of scale is 0.05 (stddev of scale change), 
    -- and a nice value for ration is a few degrees or more if your dataset allows for it

    local width = input:size(3) 
    local height = input:size(2)

    -- Scale <0=zoom in(+rand crop), >0=zoom out
    local scale_x = 0
    local scale_y = 0
    local move_x = 0
    local move_y = 0
    if augScale > 0 then
        scale_x = torch.normal(0, augScale) -- normal distribution
        -- Given a zoom in or out, we move around our canvas.
        scale_y = scale_x -- keep aspect ratio the same
        move_x = torch.uniform(-scale_x, scale_x)
        move_y = torch.uniform(-scale_y, scale_y)
    end

    -- Angle of rotation
    local rot_angle = torch.uniform(-augRot,augRot) -- (degrees) uniform distribution [-augRot : augRot)

    -- x/y grids
    local grid_x = torch.ger( torch.ones(height), torch.linspace(-1-scale_x,1+scale_x,width) )
    local grid_y = torch.ger( torch.linspace(-1-scale_y,1+scale_y,height), torch.ones(width) )

    local flow = torch.FloatTensor()
    flow:resize(2,height,width)
    flow:zero()

    -- Apply scale
    flow_scale = torch.FloatTensor()
    flow_scale:resize(2,height,width)
    flow_scale[1] = grid_y
    flow_scale[2] = grid_x
    flow_scale[1]:add(1+move_y):mul(0.5) -- move ~[-1 1] to ~[0 1]
    flow_scale[2]:add(1+move_x):mul(0.5) -- move ~[-1 1] to ~[0 1]
    flow_scale[1]:mul(height-1)
    flow_scale[2]:mul(width-1)
    flow:add(flow_scale)

    if augRot > 0 then
        -- Apply rotation through rotation matrix
        local flow_rot = torch.FloatTensor()
        flow_rot:resize(2,height,width)
        flow_rot[1] = grid_y * ((height-1)/2) * -1
        flow_rot[2] = grid_x * ((width-1)/2) * -1
        view = flow_rot:reshape(2,height*width)
        local function rmat(deg)
          local r = deg/180*math.pi
          return torch.FloatTensor{{math.cos(r), -math.sin(r)}, {math.sin(r), math.cos(r)}}
        end

        local rotmat = rmat(rot_angle)
        local flow_rotr = torch.mm(rotmat, view)
        flow_rot = flow_rot - flow_rotr:reshape( 2, height, width )
        flow:add(flow_rot)
    end

    return image.warp(input, flow, 'bilinear', false)
end

function M.ColorNormalize(meanstd)
   return function(img)
      img = img:clone()
      for i=1,3 do
         if meanstd then img[i]:add(-meanstd.mean[i]) end
         if meanstd then img[i]:div(meanstd.std[i]) end
      end
      return img
   end
end

-- Scales the smaller edge to size
function M.Scale(size, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(input)
      local w, h = input:size(3), input:size(2)
      if (w <= h and w == size) or (h <= w and h == size) then
         return input
      end
      if w < h then
         return image.scale(input, size, h/w * size, interpolation)
      else
         return image.scale(input, w/h * size, size, interpolation)
      end
   end
end

-- Crop to centered rectangle
function M.CenterCrop(size)
   return function(input)
      local w1 = math.ceil((input:size(3) - size)/2)
      local h1 = math.ceil((input:size(2) - size)/2)
      return image.crop(input, w1, h1, w1 + size, h1 + size) -- center patch
   end
end

-- Random crop form larger image with optional zero padding
function M.RandomCrop(size, padding)
   padding = padding or 0

   return function(input)
      if padding > 0 then
         local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
         temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
         input = temp
      end

      local w, h = input:size(3), input:size(2)
      if w == size and h == size then
         return input
      end

      local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
      local out = image.crop(input, x1, y1, x1 + size, y1 + size)
      assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
      return out
   end
end

-- Four corner patches and center crop from image and its horizontal reflection
function M.TenCrop(size)
   local centerCrop = M.CenterCrop(size)

   return function(input)
      local w, h = input:size(3), input:size(2)

      local output = {}
      for _, img in ipairs{input, image.hflip(input)} do
         table.insert(output, centerCrop(img))
         table.insert(output, image.crop(img, 0, 0, size, size))
         table.insert(output, image.crop(img, w-size, 0, w, size))
         table.insert(output, image.crop(img, 0, h-size, size, h))
         table.insert(output, image.crop(img, w-size, h-size, w, h))
      end

      -- View as mini-batch
      for i, img in ipairs(output) do
         output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
      end

      return input.cat(output, 1)
   end
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function M.RandomScale(minSize, maxSize)
   return function(input)
      local w, h = input:size(3), input:size(2)

      local targetSz = torch.random(minSize, maxSize)
      local targetW, targetH = targetSz, targetSz
      if w < h then
         targetH = torch.round(h / w * targetW)
      else
         targetW = torch.round(w / h * targetH)
      end

      return image.scale(input, targetW, targetH, 'bicubic')
   end
end

-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
function M.RandomSizedCrop(size)
   local scale = M.Scale(size)
   local crop = M.CenterCrop(size)

   return function(input)
      local attempt = 0
      repeat
         local area = input:size(2) * input:size(3)
         local targetArea = torch.uniform(0.08, 1.0) * area

         local aspectRatio = torch.uniform(3/4, 4/3)
         local w = torch.round(math.sqrt(targetArea * aspectRatio))
         local h = torch.round(math.sqrt(targetArea / aspectRatio))

         if torch.uniform() < 0.5 then
            w, h = h, w
         end

         if h <= input:size(2) and w <= input:size(3) then
            local y1 = torch.random(0, input:size(2) - h)
            local x1 = torch.random(0, input:size(3) - w)

            local out = image.crop(input, x1, y1, x1 + w, y1 + h)
            assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

            return image.scale(out, size, size, 'bicubic')
         end
         attempt = attempt + 1
      until attempt >= 10

      -- fallback
      return crop(scale(input))
   end
end

function M.HorizontalFlip(prob)
   return function(input)
      if torch.uniform() < prob then
         input = image.hflip(input)
      end
      return input
   end
end

function M.Rotation(deg)
   return function(input)
      if deg ~= 0 then
         input = image.rotate(input, (torch.uniform() - 0.5) * deg * math.pi / 180, 'bilinear')
      end
      return input
   end
end

-- Lighting noise (AlexNet-style PCA-based noise)
function M.Lighting(alphastd, eigval, eigvec)
   return function(input)
      if alphastd == 0 then
         return input
      end

      local alpha = torch.Tensor(3):normal(0, alphastd)
      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()

      input = input:clone()
      for i=1,3 do
         input[i]:add(rgb[i])
      end
      return input
   end
end

local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function M.Saturation(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Brightness(var)
   local gs

   return function(input)
      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Contrast(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)
      gs:fill(gs[1]:mean())

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.RandomOrder(ts)
   return function(input)
      local img = input.img or input
      local order = torch.randperm(#ts)
      for i=1,#ts do
         img = ts[order[i]](img)
      end
      return input
   end
end

function M.ColorJitter(opt)
   local brightness = opt.brightness or 0
   local contrast = opt.contrast or 0
   local saturation = opt.saturation or 0

   local ts = {}
   if brightness ~= 0 then
      table.insert(ts, M.Brightness(brightness))
   end
   if contrast ~= 0 then
      table.insert(ts, M.Contrast(contrast))
   end
   if saturation ~= 0 then
      table.insert(ts, M.Saturation(saturation))
   end

   if #ts == 0 then
      return function(input) return input end
   end

   return M.RandomOrder(ts)
end

M.collectgarbage = function()
    return function(input)
        collectgarbage(); collectgarbage();
        return input
    end
end

return M