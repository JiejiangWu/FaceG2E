import torch

def random_gamma(device,strength=0.6):
    lights = [    
            torch.tensor([strength,-0.2,0.25,-0.5,0,0,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,-0.2,0.25,0.0,0,0,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,-0.2,0.25,0.5,0,0,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,-0.4,0.25,0.0,0,0,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,0.2,0.25,0.4,0,0,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,0.2,0.25,-0.4,0,0,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,-0.2,0.25,0,-0.9,0,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,-0.2,0.25,0,-0.5,0,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,-0.2,0.25,0,0.5,0,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,-0.2,0.25,0,0.9,0,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,-0.2,0.25,0.5,0,0.5,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,-0.2,0.25,-0.5,0,0.5,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,-0.2,0.25,0.3,0,0.4,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,-0.2,0.25,-0.3,0,0.4,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,-0.2,0.25,0,0,-0.4,0,0,0] * 3)[None, ...].float().to(device),
            torch.tensor([strength,-0.2,0.25,0,0,-0.7,0,0,0] * 3)[None, ...].float().to(device),            
        ]
    return lights

