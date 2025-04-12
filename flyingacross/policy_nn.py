import torch
import torch.nn as nn
import torch.optim as optim

img_size =(16, 16)

class CNNImageEncoder(nn.Module):
    def __init__(self, image_res=(128, 128), latent_dims=128):
        super(CNNImageEncoder, self).__init__()
        self.image_res = image_res
        self.latent_dims = latent_dims
        
        # Feature extraction with stride convolutions
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

        )
        
        # self.linear_projection = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(image_res[0]//8 * image_res[1]//8 * 128, 256),
        #     nn.ELU(),
        #     nn.Linear(256, latent_dims),
        #     nn.ELU(),
        # )

        # Final projection to latent dims
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [256, 1, 1]
            # nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(128, latent_dims, kernel_size=1),  # [64, 1, 1]
            nn.Flatten()  # [64]
        )

    def forward(self, x):
        # Reshape input if needed
        if len(x.shape) == 3:  # [batch, H, W]
            x = x.unsqueeze(1)  # [batch, 1, H, W]
        
        # Forward pass
        features = self.features(x)
        latent = self.projection(features)
        return latent

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        
        self.image_encoder = CNNImageEncoder(image_res=img_size, latent_dims=128)
        
        # Original network with expanded input dimension (21 + 64 = 85)
        # self.input_projection = nn.Sequential(
        #     nn.Linear(input_dim, 64),
        #     nn.ELU(),
        # )

        self.network = nn.Sequential(
            nn.Linear(input_dim + 64, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x, depth_imgs=None):
        if depth_imgs is not None:
            # Encode depth images using VAE
            depth_latent = self.image_encoder(depth_imgs)
            x = torch.cat([x, depth_latent], dim=1)
        else:
            x = torch.cat([x, torch.zeros(x.shape[0], 64, device=x.device)], dim=1)
        return self.network(x)

class GRUPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128, latent_dims=128):
        super(GRUPolicyNetwork, self).__init__()

        
        self.image_encoder = CNNImageEncoder(image_res=img_size, latent_dims=latent_dims)
        self.input_dim = 2 * latent_dims
        self.hidden_size = hidden_size

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dims),
            nn.ELU(),
        )

        # GRU for temporal feature extraction
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=hidden_size, batch_first=True)

        # Fully connected head to output action
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ELU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, obs, depth_imgs, hidden_state=None):
        # depth_imgs: [B, 1, H, W]
        # obs: [B, D]

        depth_feat = self.image_encoder(depth_imgs)  # [B, 64]
        obs = self.input_projection(obs)
        x = torch.cat([obs, depth_feat], dim=-1).unsqueeze(1)  # [B, 1, 64+64]

        # Pass through GRU
        output, next_hidden = self.gru(x, hidden_state)  # output: [B, 1, H]
        action = self.fc(output.squeeze(1))  # [B, output_dim]
        return action, next_hidden