


import random
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, DDPMScheduler, UNet2DConditionModel, AutoencoderKL




class CLIPTokenizerWithEmbeddings(CLIPTokenizer):
    '''can read embeddings in automatic1111 format or from torch.save({placeholder_toke : embedding})'''
    
    def replace_placeholder_tokens_in_text(self, text, vector_shuffle=False):
        if hasattr(self, 'token2all_tokens'):
            for token, all_tokens in self.token2all_tokens.items():
                if vector_shuffle:
                    all_tokens = all_tokens[:]
                    random.shuffle(all_tokens)

                if isinstance(text, list):
                    text = [t.replace(token, " ".join(all_tokens)) for t in text]
                else:
                    text = text.replace(token, " ".join(all_tokens))
            print(text)
        return text
  

    def __call__(self, text, *args, vector_shuffle=False, **kwargs):
        return super().__call__(self.replace_placeholder_tokens_in_text(text, vector_shuffle=vector_shuffle), *args, **kwargs)


    def encode(self, text, *args, vector_shuffle=False, **kwargs):
        return super().encode(self.replace_placeholder_tokens_in_text(text, vector_shuffle=vector_shuffle), *args, **kwargs)



    def init_attributes(self):
        if not hasattr(self, 'token2id'):
            self.__setattr__('token2id', {})
            
        if not hasattr(self, 'token_id2all_token_ids'):
            self.__setattr__('token_id2all_token_ids', {})

        if not hasattr(self, 'token2all_tokens'):
            self.__setattr__('token2all_tokens', {})

    def load_embedding(self, token, path, text_encoder, use_orig_token=False):
        #with torch.no_grad():

            self.init_attributes()  
            
            x = torch.load(path)
            
            is_automatic1111_embedding = False
            
            if 'string_to_param' in x:
                is_automatic1111_embedding = True
                
                
            if is_automatic1111_embedding:
                # {
                #     'string_to_token' : {'*': 265},
                #     'string_to_param':tensor
                #     ...
                #  }
                
                # place_holder token will not be used in the embedding but will be replaced with the token argument
                placeholder_token = list(x['string_to_token'].keys())[0]
                embedding_matrix = x['string_to_param'][placeholder_token]
                
            else:
                placeholder_token = list(x.keys())[0]
                embedding_matrix = x[placeholder_token]
            
            if use_orig_token:
                token = placeholder_token

            print(text_encoder.get_input_embeddings().weight.data.dtype)
                
            n_vectors = embedding_matrix.shape[0]
            embedding_size = embedding_matrix.shape[1]
            
            tokens_to_add = [token] + [token+str(i) for i in range(n_vectors-1)]
            num_added_toks = self.add_tokens(tokens_to_add)
            
            text_encoder.resize_token_embeddings(self.__len__())
            
            token_id = self.convert_tokens_to_ids(token)
            token_ids = self.convert_tokens_to_ids(tokens_to_add)
            
            self.token2id[token] = token_id
            self.token_id2all_token_ids[token_id] = token_ids
            self.token2all_tokens[token] = tokens_to_add
            
            
            token_embeds = text_encoder.get_input_embeddings().weight.data
            
            token_embeds[token_id:token_id+n_vectors] = embedding_matrix
            


    def add_embedding(self, token, text_encoder, initializer_token=None, n_vectors=4, if_exists='error'):
        ''' 
        adds new token to tokenizer with multiple embeddings 
        @initializer_token: string, eg 'cat'
        @if_exists: one of ['error', 'append']
        '''

        self.init_attributes()

        if token in self.token2all_tokens: # token already exists
            if if_exists=='error':
                raise ValueError(
                  f"The tokenizer already contains the token {token}. Please pass a different"
                  " `placeholder_token` that is not already in the tokenizer."
                )
            
            existing_tokens = self.token2all_tokens[token] # eg ['<glass>', '<glass>0', '<glass>1', '<glass>2']
            tokens_to_add = [token+str(i+len(existing_tokens)-1) for i in range(n_vectors)]
            num_added_toks = self.add_tokens(tokens_to_add)

            self.token2all_tokens[token] += tokens_to_add


        else:

            tokens_to_add = [token] + [token+str(i) for i in range(n_vectors-1)]
            num_added_toks = self.add_tokens(tokens_to_add)

            self.token2all_tokens[token] = tokens_to_add


        text_encoder.resize_token_embeddings(self.__len__())

        token_ids = self.convert_tokens_to_ids(tokens_to_add)

        token_embeds = text_encoder.get_input_embeddings().weight.data

        if initializer_token:

            init_token_ids = self.encode(initializer_token, add_special_tokens=False)
            for i, id in enumerate(token_ids):
                token_embeds[id] = token_embeds[init_token_ids[i * len(init_token_ids)//n_vectors]]

        else:
            for id in token_ids:
                token_embeds[id] = torch.randn_like(token_embeds[id])



    def get_mask(self, accelerator):
        # Get the mask of the weights that won't change
        mask = torch.ones(self.__len__()).to(accelerator.device, dtype=torch.bool)

        for token in self.token2all_tokens.keys():
            token_ids = self.encode(token, add_special_tokens=False)
            for i in range(len(token_ids)):
                mask = mask & (torch.arange(self.__len__()) != token_ids[i]).to(accelerator.device)
        return mask





def save_progress(tokenizer, text_encoder, accelerator, save_path):

    for placeholder_token in tokenizer.token2all_tokens.keys():
        placeholder_token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)

        learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_ids]

        if len(placeholder_token_ids) == 1:
                learned_embeds = learned_embeds[None]
        learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
        torch.save(learned_embeds_dict, save_path)







if __name__ == '__main__':

    from transformers import CLIPTextModel, CLIPTokenizer
    import urllib.request 
    urllib.request.urlretrieve('https://huggingface.co/spaablauw/FloralMarble/resolve/main/FloralMarble-400.pt', './data/FloralMarble-400.pt')


    model_path = 'stabilityai/stable-diffusion-2-1'

    text_encoder = CLIPTextModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        #revision="fp16"
    )
    text_encoder.to(dtype=torch.float16)


    tokenizer = CLIPTokenizerWithEmbeddings.from_pretrained(model_path, subfolder="tokenizer")


    tokenizer.load_embedding('<flora-marble>', './data/FloralMarble-400.pt', text_encoder)

    # important: version 2.1 768 doesnt work with the scheduler: scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    # the 2.1-base does
    pipe = StableDiffusionPipeline.from_pretrained(model_path, tokenizer=tokenizer, text_encoder=text_encoder, safety_checker=None, torch_dtype=torch.float16).to("cuda")


    #pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    #pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    #pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    print(pipe.scheduler.config.prediction_type) # should be 'v_prediction' for 768 version


    g_cuda = torch.Generator(device='cuda')
    seed = 1965858791
    g_cuda.manual_seed(seed)


    prompt = "skulpture of greek warrior in helmet, <flora-marble>, red colors, fire"
    negative_prompt = "" 

    num_samples = 1
    guidance_scale = 7
    num_inference_steps = 50
    height = 768 #768 # 512
    width = 768  #768 # 512


    # DONT use torch.autocast("cuda") here to fix black blank images
    with torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

    for i, img in enumerate(images):
        img.save(f'./data/{i}.png')