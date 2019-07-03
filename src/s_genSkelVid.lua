--- process video and generate skeleton video
--- Generated by EmmyLua(https://github.com/EmmyLua)
--- Created by Shuangjun Liu.
--- DateTime: 3/21/2019 9:23 PM
---
require 'paths'
paths.dofile('ref.lua')
paths.dofile('model.lua')
cv = require 'cv'
require 'cv.imgproc'
require 'cv.videoio'
require 'cv.highgui'
local utils = paths.dofile('util/utils.lua')  -- image trans interface

-- user parameters
local vidNm = 'vidIR.avi'
local tarNm = 'vidIR_skel.avi'

local cap = cv.VideoCapture{filename=vidNm}
if not cap:isOpened() then
    print('failed to open video file' .. vidNm)
    os.exit(-1)
end
local ret, img = cap:read{}
local I = utils.cv2tsImg(img)
local I_pad,  idx_st, idx_end, padDrct = utils.sqrPadding(I, opt.inputRes)
local I_crop = utils.cropPadIm(I_pad, idx_st, idx_end, padDrct)
local svSize = {I_crop:size(3), I_crop:size(2)}
local videoWriter = cv.VideoWriter{
   tarNm,
   cv.VideoWriter.fourcc{'D', 'I', 'V', 'X'}, -- or any other codec
   fps = 20,   -- original is around 8 frames
   frameSize = svSize
}
local n_frms = cap:get{propId = cv.CAP_PROP_FRAME_COUNT}
local cnt =1
local hm
local out
local jt_prdt
local I_dr_skel
local I_skl_cv
print('ret number is', ret)
while ret do
    xlua.progress(cnt, n_frms)
    out = model:forward(I_pad:view(1,3,256,256):cuda())   -- batch cuda format
    hm = out[#out][1]:float() -- last output, 1st batch
    hm[hm:lt(0)] = 0        -- less than
    jt_prdt = getPreds(hm)[1]    -- local formate
    jt_prdt:mul(4)      -- scale 4 times
    -- cropped image show
    if padDrct == 2 then
        jt_prdt:select(2,1):add(-idx_st) -- get rid of how far biased from first coord
    elseif padDrct == 1 then
        jt_prdt:select(2,2):add(-idx_st)
    end
    I_crop = utils.cropPadIm(I_pad, idx_st, idx_end, padDrct)
    I_dr_skel =  drawSkeleton_demo(I_crop, hm, jt_prdt)    -- use the hm mean
    --image.display(I_dr_skel)
    I_skl_cv = utils.ts2cvImg(I_dr_skel)
    videoWriter:write{I_skl_cv}
    ret, img = cap:read{}     -- next frame upper value
    if ret then
        I = utils.cv2tsImg(img)
        I_pad,  idx_st, idx_end, padDrct = utils.sqrPadding(I, opt.inputRes)
        I_crop = utils.cropPadIm(I_pad, idx_st, idx_end, padDrct)
        cnt = cnt + 1
    end
end
