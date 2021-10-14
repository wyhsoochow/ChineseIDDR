# ChineseIDDR
如何训练：
固定两张卡进行训练
CUDA_VISIBLE_DEVICES=0,1 python main.py -func train -splitting 3
进行评测：
CUDA_VISIBLE_DEVICES=0,1 python main.py -func eval -splitting 3
用自己的数据集进行评测
首先把自己的exam.txt文件放到./data下面，两句话之间用|||隔开(已给出实例)
然后进入data文件夹,
python createHanlp.py生成input文件
CUDA_VISIBLE_DEVICES=0,1 python main.py -func test -splitting 3
