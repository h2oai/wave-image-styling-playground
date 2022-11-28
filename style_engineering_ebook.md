# Style Engineering

The StyleGAN was first used to generate images from a given domain.
This was followed by its usage for projecting real faces into a given space.

![F_4_glasses](https://user-images.githubusercontent.com/54039395/200564497-20e95d19-1de4-4bc9-8b51-cf78f919e749.png)  |  ![F_4_glasses_projected](https://user-images.githubusercontent.com/54039395/200564517-fe621f12-ef23-41a8-a90d-73e6706949cf.png)
:-------------------------:|:-------------------------:
Original Image  |  Projected Image

The projected images can then be used for a myriad of different use cases.

## Style Mixing
One can combine the features of two images were combinable using the StyleGAN.

![M_2_with_hair](https://user-images.githubusercontent.com/54039395/200565627-3cfac7e4-e497-44e1-9630-891c6385ddba.png) |  ![baby_2](https://user-images.githubusercontent.com/54039395/200565639-e9342dac-3eee-4dc9-aefd-e87e4621d69d.png) | ![M_2_with_hair,baby_2,1_8](https://user-images.githubusercontent.com/54039395/200565673-8314ade3-c117-4825-b8b5-d86350e8393d.png)
:-------------------------:|:-------------------------:|:-------------------------:
Image 1                    |  Image 2                  |  Style Mixed image with  `Z_low = 1` and `Z_high = 8`

## Image Editing
Another cool quirk of these projected images was that individual characteristics of the subject in the image were now tweakable.
![baby_1](https://user-images.githubusercontent.com/54039395/200574194-a774eb9b-cea2-471f-90f2-762b0f0f864f.png) | ![baby_1_age=2,gender=4](https://user-images.githubusercontent.com/54039395/200574116-02a94d0f-a5bc-47be-85c3-4616652d836b.png)
:-------------------------:|:-------------------------:
Original Image  |  Age = +2, Gender = +4

## Domain Adaptation
In 2020, NVIDIA released another model called StyleGAN-NADA that was capable of transfering different images to different domains.
Additionally, these results worked on top of all the features already existing in StyleGANs.

![F_1_smiling](https://user-images.githubusercontent.com/54039395/200578454-3cc96118-dbc7-4309-9925-60928728b1d4.png) | ![F_1_smiling_anime](https://user-images.githubusercontent.com/54039395/200578818-a79c498e-8a93-4f8b-819e-8e3972c667a9.png)
:-------------------------:|:-------------------------:
Original Image | Anime
![F_1_smiling_age=6](https://user-images.githubusercontent.com/54039395/200578796-ba395dbd-5947-452c-a18d-962cd34ff8d6.png) | ![F_1_smiling_anime_age=6](https://user-images.githubusercontent.com/54039395/200578808-ca5409be-f7fa-4a0e-8762-b728344a6374.png)
Age = +6 | Anime, Age = +6

## Image Restoration
In 2021, Tencent Applied Research Center developed a model called GFP-GAN, which was capable of improving image quality and resolution, specifically of images of faces.
![adele_crop](https://user-images.githubusercontent.com/54039395/200716049-56b88715-9edc-461d-b69d-5a3159e88a5e.png) | ![adele_crop_fixed](https://user-images.githubusercontent.com/54039395/200716067-f7da7a18-648d-4eb9-b94d-ce4d3d44d0f8.png)
:-------------------------:|:-------------------------:
Original Image | Restored Image


## Image Prompting
The underlying model of image generators took a quick shift from GANs to transformers with the increasing popularity of transformer-based language models in recent years.
One such algorithm that has shown impressive results is Stable Diffusion. Not only was it capable of generating images based on text prompts, but it was also capable of altering existing images using these prompts.

![M_3_glasses](https://user-images.githubusercontent.com/54039395/200601952-b1bbeda0-1e0b-40ef-9634-368a5f4898ae.png) | ![M_3_glasses_Superhero, blue glowing eyes, mask, super suit_steps=50,guidance=7 5](https://user-images.githubusercontent.com/54039395/200602228-7c68ee79-6e7d-49b5-ae3b-d62b08de6288.png) | ![Superhero, blue glowing eyes, mask, super suit_steps=50,guidance=7 5](https://user-images.githubusercontent.com/54039395/200602005-fa9cb026-e98a-41b1-aeda-f3797a63f651.png) | 
:-------------------------:|:-------------------------:|:-------------------------:
Original Image | Original image + Prompt: Superhero, blue glowing eyes, mask, super suit. Sampler=K-LMS; Steps=50; Guidance Scale=7.5 | No source image + Prompt: Superhero, blue glowing eyes, mask, super suit. Sampler=K-LMS; Steps=50; Guidance Scale=7.5

#### Image 2 Image

Seasons

![Seattle_Original](https://user-images.githubusercontent.com/1318029/204177327-dd68a342-0616-41be-8de3-f86c071c5a46.png) | ![Seattle_winter](https://user-images.githubusercontent.com/1318029/204177567-7106a676-7107-4505-8058-f6ab8751e1d3.png) | ![Seattle_summer](https://user-images.githubusercontent.com/1318029/204177672-43875b7d-0e57-417d-a0de-7d532b7e251e.png) | ![Seattle_spring](https://user-images.githubusercontent.com/1318029/204178872-ceb4a341-32de-4c41-9ee7-3a55c88a7f76.png) |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
Original Image | Seattle in Winter | Seattle in summer| Seattle in spring 
