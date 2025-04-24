import torch
from model.Model import ASD_Model

class LossAV(torch.nn.Module):
    def __init__(self):
        super(LossAV, self).__init__()
        self.FC = torch.nn.Linear(128, 2)
    
    def forward(self, x):
        return self.FC(x)[:,:,1]

class LightASD(torch.nn.Module):
    def __init__(self):
        super(LightASD, self).__init__()
        self.model = ASD_Model()
        self.lossAV = LossAV()
    
    @torch.inference_mode()
    def forward(self, video_batch, mfcc_batch):
        mfcc_embed = self.model.forward_audio_frontend(mfcc_batch)
        video_embed = self.model.forward_visual_frontend(video_batch)
        joint_embed = self.model.forward_audio_visual_backend(mfcc_embed, video_embed)
        score = self.lossAV.forward(joint_embed)
        return score