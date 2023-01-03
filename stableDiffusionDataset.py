

import random
import torch



from torch.utils.data import Dataset

import PIL

from PIL import Image

import itertools
import math
import os
import random
from pathlib import Path

from torchvision import transforms

import json 




# dreambooth:
    # single prompt: photo of xyz house
    # pick random from list of predefined prompts
    # image caption pairs
    # train multiple concepts jointly

# textual inversion
    # pick random from list of predefined prompts
    # image caption pairs
    # train multiple concepts jointly

# finetuning:
    # image caption pairs



# dreambooth collate_fn output
    # batch = {
    #     "input_ids": input_ids,
    #     "pixel_values": pixel_values,
    # }
    #     batch = {
    #         "input_ids": input_ids,
    #         "pixel_values": pixel_values,
        # }




# precedence:
# concept_list keys:
    # instance_prompt
    # instance_data_dir


    # class_prompt
    # class_data_dir


    # promptsfile     each line: {"file_name": "0001.png", "text": "This is a golden retriever playing with a ball"}
    # metadata      each line: {"file_name": "0001.png", "text": "This is a golden retriever playing with a ball"}



# if promtsfile is None, use instance_prompt
# if promtsfile and instance_prompt is None, use ramdom prompt from templates
# if token is set, must be contained in instance_prompt, if instance_prompt is not set, should be contained in promptsfile (maybe if not in all, it has regularization effect)
# for normal finetuning, only set promptsfile and instance_data_dir, everything else set to None
concepts_list = [
            {
                "token" : None, #instance_prompt must contain token (if both set), if not set, instance_prompt can contain eg rare tokens like xyz, worth trying if some prompts in promptsfile do not contain the token if it helps for preserving prior and regularization
                "initializer_token" : None, # only needed when adding "token" to embedding
                "promptsfile" : None, # precedence 1
                "instance_prompt": None, # precedence 2
                "class_prompt": None,
                "instance_data_dir":None,
                "class_data_dir": None,
                "type" : None, # one of ['style', 'object']; must be set if promptsfile and instance_prompt are not set
                
            }
        ]

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]
imagenet_style_templates_small = [
    "a photo in the style of {}",
    "a rendering in the style of {}",
    "a cropped photo in the style of {}",
    "the photo in the style of {}",
    "a clean photo in the style of {}",
    "a dirty photo in the style of {}",
    "a dark photo in the style of {}",
    "a picture in the style of {}",
    "a cool photo in the style of {}",
    "a close-up photo in the style of {}",
    "a bright photo in the style of {}",
    "a cropped photo in the style of {}",
    "a good photo in the style of {}",
    "a close-up photo in the style of {}",
    "a rendition in the style of {}",
    "a nice photo in the style of {}",
    "a small picture in the style of {}",
    "a weird picture in the style of {}",
    "a large picture in the style of {}",
]


def get_promptsfile(file_path):
    try:
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
    except Exception as e:
        print(e)
        data = []
    return data

# entry = {"file_name": filename, "text": prompt},

# imagenet_style_templates_small = [
#     "a painting in the style of {}",
#     "a rendering in the style of {}",
#     "a cropped painting in the style of {}",
#     "the painting in the style of {}",
#     "a clean painting in the style of {}",
#     "a dirty painting in the style of {}",
#     "a dark painting in the style of {}",
#     "a picture in the style of {}",
#     "a cool painting in the style of {}",
#     "a close-up painting in the style of {}",
#     "a bright painting in the style of {}",
#     "a cropped painting in the style of {}",
#     "a good painting in the style of {}",
#     "a close-up painting in the style of {}",
#     "a rendition in the style of {}",
#     "a nice painting in the style of {}",
#     "a small painting in the style of {}",
#     "a weird painting in the style of {}",
#     "a large painting in the style of {}",
# ]


class StableDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        with_prior_preservation=True,
        size=512,
        center_crop=False,
        num_class_images=None,
        pad_tokens=False,
        hflip=False,
        vector_shuffle=False
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation
        self.pad_tokens = pad_tokens

        self.vector_shuffle = vector_shuffle

        self.instance_images_path = []
        self.class_images_path = []
        self.num_class_images = num_class_images

        if isinstance(concepts_list, str):
            try:
                with open(concepts_list, "r") as f:
                    concepts_list = json.load(f)
            except Exception as e:
                raise ValueError(f"concept_list is a string and cannot be opened. if concept_list is a string, there must exist a file with this filepath.")
                  
                

        for concept in concepts_list:
            # if promtsfile is set, use promtsfile
            # if promtsfile is None, use instance_prompt
            # if promtsfile and instance_prompt are None, use ramdom prompt from templates

            if concept.get("promptsfile") is not None:
                promptsfile = get_promptsfile(concept["promptsfile"])
                inst_img_path = [(x['file_name'], x['text']) for x in promptsfile if os.path.exists(concept["instance_data_dir"].rstrip('/') + '/' +x['file_name'])]
                self.instance_images_path.extend(inst_img_path)

            elif concept.get("instance_prompt") is not None:
                if concept.get("token"): 
                    assert concept["token"] in concept["instance_prompt"], 'if "token" and "instance_prompt" are set, "instance_prompt" must contain "token".'

                inst_img_path = [(x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()] # [('./abc.png', 'photo of xyz house'), ...]
                self.instance_images_path.extend(inst_img_path)

            else:
                if concept.get('type') is None:
                    raise ValueError('if promptsfile and instance_prompt are not set in concept_list, "type" must be specified as one of ["style", "object"].')
                templates = imagenet_style_templates_small if concept['type'] == 'style' else imagenet_templates_small
                sample_prompt_from_template = lambda: random.choice(templates).format(concept["token"])
                inst_img_path = [(x, sample_prompt_from_template()) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()]
                self.instance_images_path.extend(inst_img_path)

        if num_class_images and with_prior_preservation:

            class_data_dirs =[]
            for concept in concepts_list:
                if concept["class_data_dir"] not in class_data_dirs:
                    class_img_path = [(x, concept["class_prompt"]) for x in Path(concept["class_data_dir"]).iterdir() if x.is_file()] # [('./abc.png', 'photo of house'), ...]
                    self.class_images_path.extend(class_img_path[:num_class_images])
                    class_data_dirs.append(concept["class_data_dir"])

        random.shuffle(self.instance_images_path)

        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images
        if num_class_images:
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)

        self.image_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5 * hflip),
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_path, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="max_length" if self.pad_tokens else "do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            vector_shuffle=self.vector_shuffle
        ).input_ids

        if self.num_class_images and self.with_prior_preservation:
            class_path, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_path)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                padding="max_length" if self.pad_tokens else "do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                vector_shuffle=self.vector_shuffle
            ).input_ids

        return example



def collate_fn(examples, tokenizer):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if "class_images" in examples[0]: # with with_prior_preservation
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    padded_tokens = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt",
    )

    batch = {
        "pixel_values": pixel_values,
        "input_ids": padded_tokens.input_ids,
        "attention_mask": padded_tokens.attention_mask,
    }
    return batch







class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]



class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count