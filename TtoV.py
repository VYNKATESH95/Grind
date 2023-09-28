import torch
from diffusers import AltDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = AltDiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype = torch.float16, variant = "fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "" #Enter your prompt here for generating any video
video_frames = pipe(prompt,min_interferencesteps=30),frames 
video_path = export_to_video(video_frames)
video_name = video_path.replace('/tmp','')
print('Name:', video_name)
torch.cuda.empty_cache()