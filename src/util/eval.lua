-------------------------------------------------------------------------------
-- Helpful functions for evaluation
-------------------------------------------------------------------------------
--paths.dofile('img.lua')
function loadPreds(predFile, doHm, doInp) -- saved in hdf5
    local f = hdf5.open(projectDir .. '/exp/' .. predFile .. '.h5','r')
    local inp,hms
    local idxs = f:read('idxs'):all()
    local preds = f:read('preds'):all()
    if doHm then hms = f:read('heatmaps'):all() end
    if doInp then inp = f:read('input'):all() end
    return idxs, preds, hms, inp
end

function calcDists(preds, label, normalize)
    --print('preds type is', preds:type())
    --print('label type is', label:type())
    local dists = torch.Tensor(preds:size(2), preds:size(1))    -- n_jt x nBch , normalized
    local diff = torch.Tensor(2)
    for i = 1,preds:size(1) do
        for j = 1,preds:size(2) do
            if label[i][j][1] > 1 and label[i][j][2] > 1 then
                dists[j][i] = torch.dist(label[i][j],preds[i][j])/normalize[i]
            else
                dists[j][i] = -1
            end
        end
    end
    return dists
end

function getPreds(hm) -- positive joint idx  preds  nBatch x nJoints x2
    if hm:size():size() == 3 then hm = hm:view(1, hm:size(1), hm:size(2), hm:size(3)) end
    assert(hm:size():size() == 4, 'Input must be 4-D tensor')
    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()  -- 1x1x2 same idx extend to 2
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)   -- x down scale + 1
    preds[{{}, {}, 2}]:add(-1):div(hm:size(3)):floor():add(1)   -- modulo operation,
    local predMask = max:gt(0):repeatTensor(1, 1, 2):float()    -- great than
    preds:add(-1):cmul(predMask):add(1)
    return preds
end

function distAccuracy(dists, thr)
    -- Return percentage below threshold while ignoring values with a -1
    if not thr then thr = .5 end
    if torch.ne(dists,-1):sum() > 0 then
        return dists:le(thr):eq(dists:ne(-1)):sum() / dists:ne(-1):sum()
    else
        return -1
    end
end

function heatmapAccuracy(output, label, thr, idxs)
    -- Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
    -- First value to be returned is average accuracy across 'idxs', followed by individual accuracies.  One sample acc?
    local preds = getPreds(output)
    local gt = getPreds(label)
    local dists = calcDists(preds, gt, torch.ones(preds:size(1))*opt.outputRes/10)  -- res /10 , normalization, res/10 as std, roughly normalization
    -- dists, n_jt x nBatch
    local acc = {}
    local avgAcc = 0.0
    local badIdxCount = 0

    -- DB print
    --print('current dists are', dists)   -- all negative
    --print('DB current preds are', preds)
    --print("DB read in gt are", gt)  -- gt must has problems all 1
    --print('the lable sizes are', label:size())  -- 16 x 64 x 64
    --image.display(label[1][1])

    if not idxs then
        for i = 1,dists:size(1) do
            acc[i+1] = distAccuracy(dists[i])   -- 2nd 0.5 pck
    	    if acc[i+1] >= 0 then avgAcc = avgAcc + acc[i+1]
            else badIdxCount = badIdxCount + 1 end
        end
        acc[1] = avgAcc / (dists:size(1) - badIdxCount)
    else
        for i = 1,#idxs do
            acc[i+1] = distAccuracy(dists[idxs[i]])
	    if acc[i+1] >= 0 then avgAcc = avgAcc + acc[i+1]
            else badIdxCount = badIdxCount + 1 end
        end
        if 0 == #idxs -badIdxCount then
            acc[1] = 0  -- contribute nothing to total
        else
            acc[1] = avgAcc / (#idxs - badIdxCount) -- ave all parts, each part acc
        end

    end
    return unpack(acc)
end

function basicAccuracy(output, label, thr)
    -- Calculate basic accuracy
    if not thr then thr = .5 end -- Default threshold of .5
    output = output:view(output:numel())
    label = label:view(label:numel())

    local rounded_output = torch.ceil(output - thr):typeAs(label)
    local eql = torch.eq(label,rounded_output):typeAs(label)

    return eql:sum()/output:numel()
end

function displayPCK(dists, part_idx, label, title, svFd, show_key, idx_ephz, PCKshow )
    -- Generate standard PCK plot
    -- svFd, indicate where to save the images, if false, nothing will be saved
    -- idx_ephz,  the emphasized curve
    -- history: 1. add save function
    if not (type(part_idx) == 'table') then
        part_idx = {part_idx}
    end
    PCKshow = PCKshow or 0.1
    local idx_show =math.floor( PCKshow/0.05 ) +1

    curve_res = 11
    num_curves = #dists
    local t = torch.linspace(0,.5,curve_res)    -- 0 to 0.5
    local pdj_scores = torch.zeros(num_curves, curve_res)
    local plot_args = {}
    print(title)
    print('show PCK', PCKshow)
    for curve = 1,num_curves do     -- md
        for i = 1,curve_res do      -- scores
            t[i] = (i-1)*.05
            local acc = 0.0
            for j = 1,#part_idx do
                acc = acc + distAccuracy(dists[curve][part_idx[j]], t[i])
            end
            pdj_scores[curve][i] = acc / #part_idx
        end
        --plot_args[curve] = {label[curve],t,pdj_scores[curve],'-'}   -- have all scores test part
        if idx_ephz == curve then
            plot_args[curve] = {string.format("{/:Bold %s}", label[curve]), t, pdj_scores[curve], 'lw 5 pt ' .. curve}
        else
            plot_args[curve] = {label[curve], t, pdj_scores[curve], 'lw 2 pt ' .. curve}
        end

        --print(label[curve],pdj_scores[curve][curve_res])        -- very last one shown
        print(label[curve],pdj_scores[curve][idx_show])
    end

    require 'gnuplot'
    gnuplot.raw('set title "' .. title .. '"')
    if not show_key then gnuplot.raw('unset key') 
    else gnuplot.raw('set key font ",6" outside right bottom') end
    gnuplot.raw('set xrange [0:.5]')
    gnuplot.raw('set yrange [0:1]')
    gnuplot.plot(unpack(plot_args))

    if svFd then        -- save to file
        gnuplot.raw('set terminal font "Times, 30"')
        gnuplot.pdffigure(paths.concat(svFd,title .. '.pdf'))
        gnuplot.raw('set bmargin 5 lmargin 10')
        --gnuplot.raw('set lmargin 10')
        --gnuplot.raw('set key font "Times,30" right bottom')
        if not show_key then gnuplot.raw('unset key')
         else gnuplot.raw('set key font "Times,20" outside right bottom') end    -- set 4 times bigger
        gnuplot.raw('set xrange [0:.5]')
        gnuplot.raw('set yrange [0:1]' )
        gnuplot.raw('set tics font ", 20')
        --gnuplot.xlabel('"Total PCK" font "Times,6"')
        --gnuplot.ylabel('"Estimation rate(%)" font "Times,6"')
        gnuplot.raw('set xlabel "Total PCK" font "Times,20" offset 0,-1')
        gnuplot.raw('set ylabel "Estimation rate(%)" font "Times,20" offset -2,0')
        gnuplot.plot(unpack(plot_args)) --{ {title, x, y}, {title,x,y}...} , so multiple curve
        gnuplot.plotflush()
    end
end

function drawOutput(input, hms, coords)
    -- draw skeleton and also add output
    local im = drawSkeleton_demo(input, hms, coords) -- skel image

    local colorHms = {}
    local inp64 = image.scale(input,64):mul(.3)
    for i = 1,16 do
        colorHms[i] = colorHM(hms[i])
        colorHms[i]:mul(.7):add(inp64)
    end
    local totalHm = compileImages(colorHms, 4, 4, 64)
    im = compileImages({im,totalHm}, 1, 2, 256)
    im = image.scale(im,756)
    return im
end