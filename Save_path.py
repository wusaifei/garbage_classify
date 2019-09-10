import os
dress="/media/vip/Data2/wusaifei/RSI_classification/pytorch/garbage_classify-master/datasets/garbage_classify/train_data/"
with open("train.txt","w") as f:
    for root,dirs,files in os.walk(dress):
        # root = root.replace(dress,'')
        for file in files:
            f.write(os.path.join(root, file) + "\n")
