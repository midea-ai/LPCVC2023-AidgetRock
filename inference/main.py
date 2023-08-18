from argparse import ArgumentParser, Namespace
from typing import Union, Optional, Sequence,Dict,List
import numpy as np
from process import process0
import torch 
from torch.utils.data import DataLoader
import tensorrt as trt 
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import transforms
import cv2
import os
import pkg_resources

def getArgs() -> Namespace:
    # NOTE: These variables can be changed
    programName: str = "LPCVC 2023 Sample Solution"
    authors: List[str] = ["Nicholas M. Synovic", "Ping Hu"]

    prog: str = programName
    usage: str = f"This is the {programName}"
    description: str = f"This {programName} does create a single segmentation map of arieal scenes of disaster environments captured by unmanned arieal vehicles (UAVs)"
    epilog: str = f"This {programName} was created by {''.join(authors)}"

    # NOTE: Do not change these flags
    parser: ArgumentParser = ArgumentParser(prog, usage, description, epilog)
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Filepath to an image to create a segmentation map of",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Filepath to the corresponding output segmentation map",
    )

    return parser.parse_args()

class TRTWrapper(torch.nn.Module): 
    def __init__(self,engine: Union[str, trt.ICudaEngine], 
                 output_names: Optional[Sequence[str]] = None) -> None: 
        super().__init__() 
        self.engine = engine 
        if isinstance(self.engine, str): 
            with trt.Logger() as logger, trt.Runtime(logger) as runtime: 
                with open(self.engine, mode='rb') as f: 
                    engine_bytes = f.read() 
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        else:
            # pkg_resources input io.Bytes, while runtime could only receive bytes obj.
            with trt.Logger() as logger, trt.Runtime(logger) as runtime: 
                self.engine = runtime.deserialize_cuda_engine(self.engine.getvalue())
        self.context = self.engine.create_execution_context() 
        names = [_ for _ in self.engine] 
        input_names = list(filter(self.engine.binding_is_input, names)) 
        self._input_names = input_names 
        self._output_names = output_names 
 
        if self._output_names is None: 
            output_names = list(set(names) - set(input_names)) 
            self._output_names = output_names 
 
    def forward(self, inputs: Dict[str, torch.Tensor]): 
        assert self._input_names is not None 
        assert self._output_names is not None 
        bindings = [None] * (len(self._input_names) + len(self._output_names)) 
        profile_id = 0 
        for input_name, input_tensor in inputs.items(): 
            # check if input shape is valid 
            profile = self.engine.get_profile_shape(profile_id, input_name) 
            assert input_tensor.dim() == len( 
                profile[0]), 'Input dim is different from engine profile.' 
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape, 
                                             profile[2]): 
                assert s_min <= s_input <= s_max, 'Input shape should be between ' + f'{profile[0]} and {profile[2]}' + f' but get {tuple(input_tensor.shape)}.' 
            idx = self.engine.get_binding_index(input_name) 
 
            # All input tensors must be gpu variables 
            assert 'cuda' in input_tensor.device.type 
            input_tensor = input_tensor.contiguous() 
            if input_tensor.dtype == torch.long: 
                input_tensor = input_tensor.int() 
            self.context.set_binding_shape(idx, tuple(input_tensor.shape)) 
            bindings[idx] = input_tensor.contiguous().data_ptr() 
 
        # create output tensors 
        outputs = {} 
        for output_name in self._output_names: 
            idx = self.engine.get_binding_index(output_name) 
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx)) 
 
            device = torch.device('cuda') 
            output = torch.empty(size=shape, dtype=dtype, device=device) 
            outputs[output_name] = output 
            bindings[idx] = output.data_ptr() 
        self.context.execute_async_v2(bindings, 
                                      torch.cuda.current_stream().cuda_stream) 
        return outputs 

class Mydata(Dataset):
    def __init__(self, img_path):
        super(Mydata, self).__init__()
        self.img_path = img_path
        self.images = os.listdir(self.img_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_file = os.path.join(self.img_path, self.images[index])
        image = cv2.imread(img_file)

        image = self.transform(image)
        return image, self.images[index]

    def imnormalize(self, img, mean, std, to_rgb=True):
        img = np.float32(img) if img.dtype != np.float32 else img.copy()
        return self.imnormalize_(img, mean, std, to_rgb)

    def imnormalize_(self, img, mean, std, to_rgb=True):
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img

    
    def transform(self, image):
        image = self.imnormalize(image, np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), \
         np.array([58.395, 57.12 , 57.375], dtype=np.float32))
        totensor = transforms.ToTensor()
        image = totensor(image)
        return image

SIZE: List[int] = [288, 288]

def main() -> None:
    args: Namespace = getArgs()

    modelPath = 'tiny_288_8tp_3trans_dist-682.trt'
    output_node_nema = '682'
    
    with pkg_resources.resource_stream(__name__, modelPath) as model_file:
        
        model = TRTWrapper(model_file, [output_node_nema])

        dataloader = DataLoader(Mydata(args.input),shuffle=False,batch_size=1)

        model.eval()


        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        #Warm Up
        for i, data in enumerate(dataloader):
            img, name = data
            img = img.cuda()
            img= F.interpolate(img, SIZE, None, 'bilinear', False)        
            model({'input':img})
            if(i==100): 
                break

        dataloader = DataLoader(Mydata(args.input),shuffle=False,batch_size=1)
        
        time = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                img, name = data
                name = name[0]
                img = img.cuda()
                img= F.interpolate(img, SIZE, None, 'bilinear', False)
                
                start.record()
                output = model({'input':img})
                end.record()
                torch.cuda.synchronize()
                time += start.elapsed_time(end)

                transposed = output[output_node_nema]
                transposed = F.interpolate(transposed, (512,512), None, "bilinear", False)
                out = torch.argmax(transposed, dim=1, keepdim=True)

                out = out.cpu().numpy()
                out = np.squeeze(out)
                out = process0(out)
                cv2.imwrite(os.path.join(args.output, name), out)
        
        print(time/1000)
        del transposed
        del model
        del i
        del out
        torch.cuda.empty_cache()
        model_file.close()
