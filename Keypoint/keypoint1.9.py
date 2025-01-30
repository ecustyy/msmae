import cv2
import numpy as np
import os

saveroot = r'F:\task9\me\mmc-amd-main2.1\output\cfp-mask-data.npz'
f = np.load(saveroot, allow_pickle=True)
masks = f['arr_0'] #mask
# features = f['arr_1']   #feature

refeats = []
data_dir = "E:/task8/GAMMA/training_data/multi-modality_images"
cfplist = os.listdir(data_dir)  # 读取子文件夹下的所有文件
for ic,cfp in enumerate(cfplist):
    cfpdir = os.path.join(data_dir, cfp, cfp+'.jpg')
    img = cv2.imread(cfpdir)
    if img.shape[0] == 2000:
        img = img[1000 - 967:1000 + 967, 1496 - 978:1496 + 978, :]  # 裁剪过大图片
    img = cv2.resize(img,dsize=(224,224),interpolation=cv2.INTER_LINEAR)

    # impath = os.path.join(r'CFPkp4\ori', cfp + '.png')
    # # cv2.imshow('drawKeypoints', img2)
    # cv2.imwrite(impath, img)
    cfp2dir = os.path.join(r'F:\task9\me\mmc-amd-main2.1\output', cfp+'.png')
    img2 = cv2.imread(cfp2dir)
    if img2.shape[0] == 2000:
        img2 = img2[1000 - 967:1000 + 967, 1496 - 978:1496 + 978, :]  # 裁剪过大图片
    img2 = cv2.resize(img2,dsize=(224,224),interpolation=cv2.INTER_LINEAR)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)
    ret, imgb = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('imgb', imgb)
    # cv2.waitKey(0)
    kernel = np.ones((5, 5), np.uint8)
    imgd = cv2.dilate(imgb, kernel, iterations=1)
    # cv2.imshow('j', imgd)
    # cv2.waitKey(0)
    kernel = np.ones((10, 10), np.uint8)
    imge = cv2.erode(imgd, kernel, iterations=5)
    # cv2.imshow('j', imge)
    # cv2.waitKey(0)
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.01) # 定义sift对象 nfeatures是未加mask的所有点个数， contrastThreshold用来约束选取点的response，
    #可以先设置一个很小的值，得到很多的关键点，按response来排序，重新组成列表kp
    #提取mask，用于关键点检测
    mask = masks[ic].squeeze()  # mask
    maskd = mask>0.2
    maskd = maskd.astype('uint8')
    kp = sift.detect(img, imge)         # 关键点集合 .pt关键点位置坐标 .size关键点邻域直径 , maskd

    kpi = []
    kpo = []
    for i in range(len(kp)):
        pt = np.round(kp[i].pt).astype(int)
        if maskd[pt[1], pt[0]]:
            kpi.append(kp[i])
        else:
            kpo.append(kp[i])

    #提取response
    reslist = np.zeros(len(kpi))
    for i in range(len(kpi)):
        reslist[i] = kpi[i].response
    #按response排序，选取前几个kp
    idx = np.argsort(reslist)
    kp2 = []
    for ik in range(20):
        try:
            kp2.append(kpi[idx[-ik-1]]) #idx是由小到大，倒序
        except:
            break
    # 可视化筛选的关键点
    for ki in kp2:
        clr = np.random.randint(0,255,(1,3))
        clr_tup =  tuple(map(int, clr[0]))
        cv2.drawKeypoints(img2, [ki], img2, color=clr_tup, flags=5)
    impath = os.path.join(r'CFPkp6\1', cfp + '.png')
    # cv2.imshow('drawKeypoints', img2)
    cv2.imwrite(impath, img2)
    # 继续排序
    kp3 = []
    for ki in range(ik+1,len(kp)):
        try:
            kp3.append(kp[idx[-ki-1]]) #idx是由小到大，倒序
        except:
            break
    # 可视化mask内所有关键点
    for ki in kp3:
        clr = np.random.randint(0, 255, (1, 3))
        clr_tup = tuple(map(int, clr[0]))
        cv2.drawKeypoints(img2, [ki], img2, color=clr_tup, flags=5)
    impath = os.path.join(r'CFPkp6\2', cfp +'.png')
    cv2.imwrite(impath, img2)

   # 可视化所有关键点
    for ki in kpo:
        clr = np.random.randint(0, 255, (1, 3))
        clr_tup = tuple(map(int, clr[0]))
        cv2.drawKeypoints(img2, [ki], img2, color=clr_tup, flags=5)
    impath = os.path.join(r'CFPkp6\3', cfp +'.png')
    cv2.imwrite(impath, img2)