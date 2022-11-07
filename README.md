# H2O Image Styling Art Studio
Style images and have fun. 
Currently, the Art Studio currently supports the following options for image generation,
1. **Image restoration:** using GFP-GAN for face restoration
2. **Image Styling:** Using StyleGAN2 with Adaptive Discriminator Augmentation
3. **Image Editing:** For facial images using landmark detection for image alginment and editing StyleGAN2 latent space
4. **Image Prompt:** Create art using Text-to-Image generation with Stable Diffusion


## Features
- Image styling
<img width="1200" alt="" src="https://user-images.githubusercontent.com/1318029/174354513-cf58d0d2-d1dd-4d86-a95a-e73c1bc05c53.png">

- Image editing
![image_editing](https://user-images.githubusercontent.com/1318029/174355648-dbfee284-305c-4707-85e9-14fd9c1e4b97.gif)

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


## More faces
Find more faces [here](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL) or upload your own and have fun


## References/Credits
1. https://github.com/TencentARC/GFPGAN
```
@InProceedings{wang2021gfpgan,
    author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
    title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}
```
2. https://github.com/NVlabs/stylegan2-ada-pytorch
```
@inproceedings{Karras2020ada,
  title     = {Training Generative Adversarial Networks with Limited Data},
  author    = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
  booktitle = {Proc. NeurIPS},
  year      = {2020}
}
```
3. Latent Space Exploration with StyleGAN2: https://github.com/AmarSaini/Epoching-Blog/blob/master/_notebooks/2020-08-10-Latent-Space-Exploration-with-StyleGAN2.ipynb
4. https://github.com/huggingface/diffusers
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
