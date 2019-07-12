import os
dsFd = 'G:\My Drive\ACLab Shared GDrive\datasetPM'
lbNm = 'danaLab'
covLs = {'uncover', 'cover1', 'cover2'}
mdLs = {'RGB','IR'}
# mdNm ='IR'
for i in range(30):
    subjNm = '{:05}'.format(i+1)
    for mdNm in mdLs:
        for covNm in covLs:
            covFd = os.path.join(dsFd, lbNm, subjNm, mdNm, covNm)
            files = os.listdir(covFd)
            print('subj {} modal {} cov {} has files {}'.format(i+1, mdNm, covNm, len(files)))
            if len(files) < 45:
                print('file missing alert')