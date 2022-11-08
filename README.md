# H2O Image Styling Art Studio
Style images and have fun.

Web app built using H2O <img width="80" alt="logo" src="https://user-images.githubusercontent.com/1318029/200458962-4de45eb3-2403-4fe6-b76c-1b92adf31a2c.png"> [SDK](https://github.com/h2oai/wave), Realtime Web Apps and Dashboards for Python and R. 
Currently, the Art Studio supports the following options for image generation,
1. **Image restoration:** using GFP-GAN for face restoration
2. **Image Styling:** Using StyleGAN2 with Adaptive Discriminator Augmentation
3. **Image Editing:** For facial images using landmark detection for image alginment and editing StyleGAN2 latent space
4. **Image Prompt:** Create art using Text-to-Image generation with Stable Diffusion


## Features
- Ability to upload an image and detect emotion
<table>
  <tr>
     <td>Neutral</td>
     <td>Happy</td>
     <td>Sad</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/1318029/174356894-e0d3633f-0e0d-4317-ad0e-ce9981bf167a.png" width=270 height=480></td>
    <td><img src="https://user-images.githubusercontent.com/1318029/174357382-0c9a4cbe-9680-4266-8445-81afc838f552.png" width=270 height=480></td>
    <td><img src="https://user-images.githubusercontent.com/1318029/174359514-26317b6f-5e1c-4f17-acc7-4ede68aafe6e.png" width=270 height=480></td>
  </tr>
 </table>

- Image Restoration
<img width="1665" alt="Screen Shot 2022-11-07 at 2 52 11 PM" src="https://user-images.githubusercontent.com/1318029/200434486-29e6b9d6-9ee7-4568-b9bb-465bcc66d36f.png">

- Image styling
<img width="1676" alt="Screen Shot 2022-11-07 at 3 22 29 PM" src="https://user-images.githubusercontent.com/1318029/200436526-6c8fb1c7-2b48-4415-822f-8b14a353e405.png">

- Image editing
<img width="1669" alt="Screen Shot 2022-11-07 at 3 30 31 PM" src="https://user-images.githubusercontent.com/1318029/200437512-fdd8c01b-f9de-4d97-b136-f53343dbba01.png">

![M_3_glasses_fun](https://user-images.githubusercontent.com/1318029/200437343-16c6238f-59e8-4f04-806f-1e2b64870d38.gif)

- Image Prompt
<img width="1667" alt="Screen Shot 2022-11-07 at 3 47 56 PM" src="https://user-images.githubusercontent.com/1318029/200439500-4bb88901-ce68-44d9-b10b-9c6a5099b008.png">


## More faces
Find more faces [here](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL) or upload your own and have fun


## References/Credits
1. GFPGAN: https://github.com/TencentARC/GFPGAN
```
@InProceedings{wang2021gfpgan,
    author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
    title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}
```
2. StyleGAN2-ADA: https://github.com/NVlabs/stylegan2-ada-pytorch
```
@inproceedings{Karras2020ada,
  title     = {Training Generative Adversarial Networks with Limited Data},
  author    = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
  booktitle = {Proc. NeurIPS},
  year      = {2020}
}
```
3. Latent Space Exploration with StyleGAN2: https://github.com/AmarSaini/Epoching-Blog/blob/master/_notebooks/2020-08-10-Latent-Space-Exploration-with-StyleGAN2.ipynb
4. Latent Diffusion Models: https://arxiv.org/abs/2112.10752

https://github.com/huggingface/diffusers; 
https://github.com/CompVis/stable-diffusion; 
https://github.com/runwayml/stable-diffusion; 
https://stability.ai/blog/stable-diffusion-public-release; 
https://github.com/LAION-AI/
```
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```
