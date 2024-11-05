import torch, einops
from torch import nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Takes (batch size, num frames, height, width, num channels) as input -> returns (batch size, number of tubes, 1D representation of tube)
class VidInputEmbedding(nn.Module):
    def __init__(self, tube_hw=16, tube_d = 4, n_channels=3, device=device, latent_size=768, batch_size=8):
        super(VidInputEmbedding, self).__init__()
        self.latent_size = latent_size
        self.tube_hw = tube_hw
        self.tube_d = tube_d
        self.n_channels=n_channels
        self.device = device
        self.batch_size = batch_size
        self.input_size = self.tube_hw*self.tube_hw*self.tube_d*self.n_channels
        
        #linear projection 
        self.linearProjection = nn.Linear(self.input_size, self.latent_size) # Does output = input @ weight + biases, @ is a dot product. If input tensor is [batch_size, height, width], 
                                                                            # "input" is considered to be shape [height (input_size in call),width], 
                                                                            # 'weight' is [width, output_size (latent_size in call)], and
                                                                            # 'biases' is [width, output_size]. The same 'weight' and 'biases' are applied to every batch element
        
        #Class Token appended used for supervised learning somehow??
        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)
        #Positional Embedding adds embedding based on position of patch in image
        self.pos_embedding = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)
    
    def forward(self,input_data):
        input_data = input_data.to(self.device)
        if self.n_channels == 3:
            patches = einops.rearrange(
                input_data, 'b f (h h1) (w w1) c -> b (h w) (f h1 w1 c)', h1 =self.tube_hw, w1 = self.tube_hw)
        else:
            patches = einops.rearrange(
                input_data, 'b f (h h1) (w w1) -> b (h w) (f h1 w1)', h1 =self.tube_hw, w1 = self.tube_hw)
        
        #print(input_data.size())
        #print(patches.shape)
        
        #Reduce patch size from patch_height_size * patch_width_size * num_channels to latent_dim through a mlp layer (A.K.A. learned linear transformation)
        linear_projection = self.linearProjection(patches).to(self.device)
        b,n,_ = linear_projection.shape
        #print(linear_projection.shape)
        # Append classification token to linear projection 
        linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
        
        #Reshape positional embedding to fit new size of linear projection
        pos_embed = einops.repeat(self.pos_embedding, 'b 1 d -> b m d', m = n+1) # d = one flat patch size but is random numbers
                                                                                # b = batch size
                                                                                # this essentially takes d and repeats it row wise to make it so d is repeated m times on top of each other i.e. w/o batch size [d0,d1,d2,d3,...,dm]'
        #print(linear_projection.size())
        #print(pos_embed.size())
        
        linear_projection += pos_embed
        return linear_projection
 

class VidEncoderBlock(nn.Module):
    def __init__(self, device=device, latent_size=768, num_heads=12,dropout=0.1):
        super(VidEncoderBlock, self).__init__()
        self.latent_size = latent_size
        self.device = device
        self.num_heads = num_heads
        self.dropout = dropout
        
        #Normalization layer
        self.norm =nn.LayerNorm(self.latent_size)
        
        #MHA layer
        self.multihead = nn.MultiheadAttention(self.latent_size, self.num_heads, dropout =self.dropout)
        
        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size *4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size*4, self.latent_size),
            nn.Dropout(self.dropout)
        )
    
    def forward(self,embedded_patches):
        firstnorm_out = self.norm(embedded_patches)
        attention_out = self.multihead(firstnorm_out, firstnorm_out, firstnorm_out)[0]
        
        # First residual connection
        first_added = embedded_patches + attention_out
        
        secondnorm_out = self.norm(first_added)
        ff_out = self.enc_MLP(secondnorm_out)
        
        return ff_out + first_added
    
   
class ViVit(nn.Module):
    def __init__(self, num_encoders=12, latent_size=768, device=device, num_heads=12, num_class=2, dropout=0.1, tube_hw=16, tube_d = 4, n_channels=3, batch_size=8):
        super(ViVit,self).__init__()
        self.num_encoders = num_encoders
        self.latent_size = latent_size
        self.device = device
        self.num_class = num_class
        self.dropout = dropout
        
        self.embedding = VidInputEmbedding(tube_hw, tube_d, n_channels, device, latent_size, batch_size)
        
        self.encStack = nn.ModuleList([VidEncoderBlock(device, latent_size, num_heads, dropout) for i in range(self.num_encoders)])
        
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.Linear(self.latent_size, self.num_class)
        )
        
    def forward(self,input):
        enc_output = self.embedding(input)
        for enc_layer in self.encStack:
            enc_output = enc_layer(enc_output)
        
        cls_token_embed = enc_output[:,0]
        
        return self.MLP_head(cls_token_embed)