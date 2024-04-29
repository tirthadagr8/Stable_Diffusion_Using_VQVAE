## this function is used to retrieve encoded tokenized text data
from sd.parameters import hp
import torch
from transformers import DistilBertModel, DistilBertTokenizer, CLIPTokenizer, CLIPTextModel

def get_tokenizer_and_model(model_type, device, eval_mode=True):
    assert model_type in ('bert', 'clip'), "Text model can only be one of clip or bert"
    if model_type == 'bert':
        text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        text_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    else:
        text_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
        text_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16').to(device)
    if eval_mode:   
        text_model.eval()
    return text_tokenizer, text_model
   

# data_transforms = T.Compose([
#     T.Resize(im_size),
#     T.CenterCrop(im_size),
#     T.ToTensor(),
#     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

class Util:
    def __init__(self,model:str):
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.text_tokenizer,self.text_model=get_tokenizer_and_model(model,self.device)
        

    def get_text_representation(self,text,
                                truncation=True,
                                padding='max_length',
                                max_length=77):
        token_output = self.text_tokenizer(text,
                                      truncation=truncation,
                                      padding=padding,
                                      return_attention_mask=True,
                                      max_length=max_length)
        indexed_tokens = token_output['input_ids']
        att_masks = token_output['attention_mask']
        tokens_tensor = torch.tensor(indexed_tokens).to(self.device)
        mask_tensor = torch.tensor(att_masks).to(self.device)
        text_embed = self.text_model(tokens_tensor, attention_mask=mask_tensor).last_hidden_state
        return text_embed
        

    def get_time_embedding(self,time_steps, temb_dim):
        """
        Convert time steps tensor into an embedding using the
        sinusoidal time embedding formula
        :param time_steps: 1D tensor of length batch size
        :param temb_dim: Dimension of the embedding
        :return: BxD embedding representation of B time steps
        """
        assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
        # factor = 10000^(2i/d_model)
        factor = 10000 ** ((torch.arange(
            start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
        )
    
        # pos / factor
        # timesteps B -> B, 1 -> B, temb_dim
        t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        return t_emb

util=Util(hp.text_model)