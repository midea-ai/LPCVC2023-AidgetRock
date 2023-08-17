import os

#训练

# config = 'local_configs/topformer/topformer_tiny_512x512_160k_2x8_ade20k.py'
# work_dir='topformer/tiny_bs16_loss18ce'

# instruction = 'python tools/train.py '+config+' --work-dir '+work_dir
# print(instruction)


#测试
path='/data/private/TopFormer/topformers/tiny_384_all_8-5'

list = os.listdir(path)
for filename in list:
    if filename[-1]=='y':
        config=os.path.join(path, filename)
        break

#config = '/data/private/TopFormer/local_configs/for_onnx/topformer_tiny_160_160k_2x8_ade20k.py'

for filename in list:
    if filename[-1]=='h' and filename=='iter_60000.pth':
        file = os.path.join(path, filename)
        instruction = 'python /data/private/TopFormer/tools/mytest.py '+config+' --checkpoint '+file+' --eval mDice'
        os.system(instruction)
        print(filename)
        print()