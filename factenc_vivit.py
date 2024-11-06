import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

# Takes (batch size, num frames, height, width, num channels) as input -> returns (batch size, number of tubes, 1D representation of tube)
class VidInputEmbedding(nn.Module):
    def __init__(self, patch_size=16, num_frames=4, n_channels=3, device='cuda', latent_size=768, batch_size=8, image_size=128):
        super(VidInputEmbedding, self).__init__()
        self.latent_size = latent_size      # D: dimension of embeddings
        self.patch_size = patch_size        # P: size of each patch 
        self.num_frames = num_frames        # F: number of frames
        self.n_channels = n_channels        # C: number of channels
        self.device = device
        self.batch_size = batch_size        # B: Batch size
        self.num_patch = (image_size//patch_size)**2    # N: number of patches per frame
        self.input_size = self.patch_size * self.patch_size * self.n_channels   # Size of flattened patch
        
        # Linear projection from patch size to latent size
        # in: (P * P * C) -> out: D
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)
        
        # Class token to be prepended to each frame
        # Shape: [1, 1, D]
        self.class_token = nn.Parameter(torch.randn(1, 1, self.latent_size)).to(device)
        
        # Positional embedding added to patches + cls token
        # Shape: [1, 1, N+1, D] where N+1 accounts for cls token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1, self.num_patch + 1, self.latent_size)
            ).to(device)
    
    def forward(self, input_data):
        # input data shape: [B, F, H, W, C]
        # where H=W=image_size
        input_data = input_data.to(self.device)

        if self.n_channels == 3:
            # Rearrange into patches
            # From: [B, F, H, W, C]
            # To: [B, F, (H/P)*(W/P), P*P*C]
            patches = einops.rearrange(
                input_data, 'b f (h h1) (w w1) c -> b f (h w) (h1 w1 c)',
                h1=self.patch_size, w1=self.patch_size
            )
        else:
            # Rearrange into patches
            # From: [B, F, H, W]
            # To: [B, F, (H/P)*(W/P), P*P]
            patches = einops.rearrange(
                input_data, 'b f (h h1) (w w1) -> b f (h w) (h1 w1)',
                h1=self.patch_size, w1=self.patch_size
            )
        # Project patches to latent dimension
        # From: [B, F, N, P*P*C] -> [B, F, N, D]
        linear_projection = self.linearProjection(patches)
        
        batch_size,num_frames,num_patches,_= linear_projection.shape
        
        # Create and append class tokens
        # class_tokens shape: [B, F, 1, D]
        class_tokens = self.class_token.expand(batch_size, num_frames, 1, self.latent_size)
        
        # Concatenate to get [B, F, N+1, D] (adds one D dimensional token per frame)
        linear_projection = torch.cat((class_tokens, linear_projection), dim=2)
        
        # Add positional embedding (Pos embeddings are an offset of existing values not additional tokens)
        # pos_embedding expands from [1, 1, N+1, D] to [B, F, N+1, D] (add same offset to patch in same spatial location irrelevant of frame)
        linear_projection += self.pos_embedding.expand(batch_size, num_frames, -1, -1)
        
        return linear_projection

class TemporalEmbedding(nn.Module):
    def __init__(self, num_frames, latent_size, num_patches, device='cuda'):
        super(TemporalEmbedding, self).__init__()
        
        # Temporal positional embedding
        self.temporal_pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, 1,latent_size)
        ).to(device)
        
        # IF I move CLS token before embedding [B, N, D]
        # self.temporal_pos_embedding = nn.Parameter(
        #     torch.randn(1, num_frames, latent_size)
        # ).to(device)
    
    def forward(self, x):
        # Add temporal positional embedding to x
        return x + self.temporal_pos_embedding.expand(x.shape[0], -1, x.shape[2], -1)

class SpatialTransformerEncoder(nn.Module):
    def __init__(self, latent_size=768, num_heads=12, dropout=0.1):
        super(SpatialTransformerEncoder, self).__init__()
        self.latent_size = latent_size
        
        # For processing patches within a single frame
        self.norm1 = nn.LayerNorm(latent_size)
        self.attn = nn.MultiheadAttention(latent_size, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(latent_size)
        self.ffn = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size * 4, latent_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x shape: [p (b*f) d] where:
        # b*f = batch_size * num_frames (treating each frame independently)
        # p = num_patches + 1 (patches + class token)
        # d = latent_size
        
        # Self-attention among patches
        norm_x = self.norm1(x)
        attn_outputs, attn_weights = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_outputs
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x, attn_weights


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, latent_size=768, num_heads=12, dropout=0.1):
        super(TemporalTransformerEncoder, self).__init__()
        self.latent_size = latent_size
        
        # For processing temporal realtionships
        self.norm1 = nn.LayerNorm(latent_size)
        self.attn = nn.MultiheadAttention(latent_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(latent_size)
        self.ffn = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size * 4, latent_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x shape: [b f p d] where:
        # b = batch_size
        # f = num_frames
        # p = num_patches + 1 (patches + class token)
        # d = latent_size
        #batch_size, num_frames, num_patches, d = x.shape
    
        
        # Self-attention across frames
        norm_x = self.norm1(x)
        attn_outputs, attn_weights = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_outputs
        
        # FFN
        x = x + self.ffn(self.norm2(x))
    
        return x, attn_weights

class ViVit(nn.Module):
    def __init__(self, num_spatial_encoders=12, num_temporal_encoders=4, latent_size=768, 
                 device='cuda', num_heads=12, num_class=2, dropout=0.1, patch_size=16, 
                 num_frames=4, n_channels=3, batch_size=8, image_size = 128):
        super(ViVit, self).__init__()
        self.num_spatial_encoders = num_spatial_encoders        # Number of encoders for spatial encoding
        self.num_temporal_encoders = num_temporal_encoders      # Number of encoders for temporal encoding
        self.latent_size = latent_size                          # Dimension of qkv
        self.device = device                                    # Device model is on
        self.num_class = num_class                              # Numer of Classes (Binary so only 2)
        self.dropout = dropout                                  # Amount of dropout to regulate overfitting
        self.batch_size = batch_size                            # Batch Size for reference
        self.num_patches = (image_size//patch_size)**2          # Total number of patches
        
        # Embeds Video Spatially (i.e. vectorizes frames and adds positional embedding and classification tokens)
        self.embedding = VidInputEmbedding(patch_size, num_frames, n_channels, device, latent_size, batch_size,image_size)
        
        # Adds a temporal positionl embedding token 
        self.temporal_embedding = TemporalEmbedding(num_frames, latent_size, self.num_patches, device)
        
        # Creates the encoder stacks
        self.spatial_encStack = nn.ModuleList([SpatialTransformerEncoder(latent_size, num_heads, dropout) for _ in range(self.num_spatial_encoders)])
        self.temporal_encStack = nn.ModuleList([TemporalTransformerEncoder(latent_size, num_heads, dropout) for _ in range(self.num_temporal_encoders)])
        
        # Final classification
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(latent_size), # should be latent_size * num_frames if doing all class tokens + patches
            nn.Dropout(dropout),
            nn.Linear(latent_size, latent_size// 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size // 2, num_class)
        )
        
    def forward(self, input):
        # Spatial Processing
        enc_output = self.embedding(input) #[B,F,P+1,D]
        
        #Ensure each frame is processed independently for spatial relationships within a frame (VIT essentially)
        enc_output = einops.rearrange(enc_output, 'b f p d -> (b f) p d')
        
        # Spatial attention - maintains all patch information
        for enc_layer in self.spatial_encStack:
            enc_output, enc_weights = enc_layer(enc_output)
        
        # Restore frame dimension with all spatial information
        enc_output = einops.rearrange(enc_output,'(b f) p d -> b f p d', b = self.batch_size)  # Separate batch and frames
        
        # Add temporal embedding
        enc_output = self.temporal_embedding(enc_output)
        
        #enc_output = einops.rearrange(enc_output,'b f d -> f b d') 
        
        
        # Reshape to make frames attend to each other
        # First, make frames the sequence dimension
        # Then, merge batch and patches to treat each patch position across time as a sequence
        enc_output = einops.rearrange(enc_output, 'b f p d -> (b p) f d')
        
        # Temporal attention process ALL spatial tokens
        for enc_layer in self.temporal_encStack:
            enc_output, enc_weights = enc_layer(enc_output)
        
        enc_output = einops.rearrange(enc_output, '(b p) f d -> b f p d', b =self.batch_size)
            
        cls_tokens = enc_output[:, :, 0, :]
            
        #enc_output = einops.rearrange(enc_output, 'f b d -> b f d')
            
        # Restore original shape: [b f p d]
        # x = x.transpose(0, 1)
        # x = einops.rearrange(x, 'f (p b) d -> b f p d', b=self.batch_size)
        
        # Classification token
        
        sequence_repr = torch.mean(cls_tokens, dim=1)
        
        # # Compute attention weights for each frame
        # attention_logits = self.sequence_attention(cls_tokens)  # [B, 8, 1]
        # attention_weights = F.softmax(attention_logits, dim=1)  # Normalize across frames
        
        # # Weighted combination of class tokens
        # sequence_repr = (cls_tokens * attention_weights).sum(dim=1)  # [B, D]

        # Classification
        output = self.MLP_head(sequence_repr)
        
        return output
    
class ViVit_2(nn.Module):
    def __init__(self, num_spatial_encoders=12, num_temporal_encoders=4, latent_size=768, device='cuda', num_heads=12, num_class=2, dropout=0.1, patch_size=16, num_frames=4, n_channels=3, batch_size=8, image_size = 128):
        super(ViVit_2, self).__init__()
        self.num_spatial_encoders = num_spatial_encoders
        self.num_temporal_encoders = num_temporal_encoders
        self.latent_size = latent_size
        self.device = device
        self.num_class = num_class
        self.dropout = dropout
        self.batch_size = batch_size
        
        self.embedding = VidInputEmbedding(patch_size, num_frames, n_channels, device, latent_size, batch_size,image_size)
        self.temporal_embedding = TemporalEmbedding(num_frames, latent_size, device)
        
        self.spatial_encStack = nn.ModuleList([SpatialTransformerEncoder(latent_size, num_heads, dropout) for _ in range(self.num_spatial_encoders)])
        self.temporal_encStack = nn.ModuleList([TemporalTransformerEncoder(latent_size, num_heads, dropout) for _ in range(self.num_temporal_encoders)])

        self.MLP_adapter = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size * 4, latent_size),
            nn.Dropout(dropout)
        )
        
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.num_class)
        )
        
    def forward(self, input):
        enc_output = self.embedding(input)
        
        #Ensure each frame is processed independently for spatial
        enc_output = einops.rearrange(enc_output, 'b f p d -> (b f) p d')
        
        # Pass through spatial encoders
        for enc_layer in self.spatial_encStack:
            enc_output, enc_weights = enc_layer(enc_output)
        
        enc_output = einops.rearrange(enc_output,'(b f) p d -> b f p d', b = self.batch_size)  # Separate batch and frames
        
        # Average pooling over spatial tokens
        enc_output = enc_output.mean(dim=2)
        
        enc_output = enc_output + self.MLP_adapter(enc_output)
        
        # Add temporal embedding
        enc_output = self.temporal_embedding(enc_output)
        
        # Pass through temporal encoders
        for enc_layer in self.temporal_encStack:
            enc_output, enc_weights = enc_layer(enc_output)
        
        # Classification token
        cls_token_embed = enc_output[:, 0]
        
        return self.MLP_head(cls_token_embed)

class SemiCon_ViVit(nn.Module):
    def __init__(self, num_spatial_encoders=12, depth=2, num_temporal_encoders=4, latent_size=768, device='cuda', num_heads=12, num_class=2, dropout=0.1, patch_size=16, num_frames=4, n_channels=3, batch_size=8, image_size = 128):
        super(SemiCon_ViVit, self).__init__()
        self.num_spatial_encoders = num_spatial_encoders
        self.num_temporal_encoders = num_temporal_encoders
        self.latent_size = latent_size
        self.device = device
        self.num_class = num_class
        self.dropout = dropout
        self.batch_size = batch_size
        self.depth = depth
        
        self.embedding = VidInputEmbedding(patch_size, num_frames, n_channels, device, latent_size, batch_size,image_size)
        self.temporal_embedding = TemporalEmbedding(num_frames, latent_size, device)
        
        self.spatial_encStack = nn.ModuleDict({str(layer_id):SpatialTransformerEncoder(latent_size, num_heads, dropout) for layer_id in range(self.num_spatial_encoders)})
        self.temporal_encStack = nn.ModuleDict({str(layer_id): TemporalTransformerEncoder(latent_size, num_heads, dropout) for layer_id in range(self.num_temporal_encoders)})
        self.spatial_crossattn = nn.ModuleDict({str(layer_id): nn.MultiheadAttention(latent_size, num_heads, dropout=dropout) for layer_id in range(1, self.depth)})
        self.temporal_crossattn = nn.ModuleDict({str(layer_id): nn.MultiheadAttention(latent_size, num_heads, dropout=dropout) for layer_id in range(1, self.depth)})
        
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.num_class)
        )

    def forward(self, input):
        enc_output = self.embedding(input)
        
        #Ensure each frame is processed independently for spatial
        enc_output = einops.rearrange(enc_output, 'b f p d -> (b f) p d')
        outputs = [enc_output] * self.depth
        
        # Pass through spatial encoders
        for layer_id in range(0, self.num_spatial_encoders, self.depth):
            if layer_id == 0:
                for i in range(self.depth):
                    outputs[i], _ = self.spatial_encStack[str(layer_id + i)](outputs[i])
            else:
                for i in range(self.depth):
                    if i ==0:
                        outputs[i], _ = self.spatial_encStack[str(layer_id + i)](outputs[i])
                    else: 
                        outputs[i], _ = self.spatial_encStack[str(layer_id + i)](outputs[i] + outputs[i-1])
                        
        for i in reversed(range(1, self.depth)):
            if i == self.depth-1:
                attended_output, _ = self.spatial_crossattn[str(i)](
                    query=outputs[i-1],
                    key=outputs[i],
                    value=outputs[i]
                )
            else:
                attended_output, _ = self.spatial_crossattn[str(i)](
                    query=outputs[i-1],
                    key=attended_output,
                    value =attended_output
                )
                        
        enc_output = einops.rearrange(attended_output,'(b f) p d -> b f p d', b = self.batch_size)  # Separate batch and frames
        
        # Average pooling over spatial tokens
        enc_output = enc_output.mean(dim=2)
        
        # Add temporal embedding
        outputs = [self.temporal_embedding(enc_output)] * self.depth
        
        for layer_id in range(0, self.num_temporal_encoders, self.depth):
            if layer_id == 0:
                for i in range(self.depth):
                    outputs[i], _ = self.temporal_encStack[str(layer_id + i)](outputs[i])
            else:
                for i in range(self.depth):
                    if i ==0:
                        outputs[i], _ = self.temporal_encStack[str(layer_id + i)](outputs[i])
                    else: 
                        outputs[i], _ = self.temporal_encStack[str(layer_id + i)](outputs[i] + outputs[i-1])
        
        for i in reversed(range(1, self.depth)):
            if i == self.depth-1:
                attended_output, _ = self.temporal_crossattn[str(i)](
                    query=outputs[i-1],
                    key=outputs[i],
                    value=outputs[i]
                )
            else:
                attended_output, _ = self.temporal_crossattn[str(i)](
                    query=outputs[i-1],
                    key=attended_output,
                    value =attended_output
                )
        del outputs
        # Classification token
        cls_token_embed = attended_output[:, 0]
        #print(cls_token_embed)
        
        return self.MLP_head(cls_token_embed)