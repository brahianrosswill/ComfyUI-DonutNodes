This is an experimental node I made to mimick the "adjust" in A1111 Supermerger (https://github.com/hako-mikan/sd-webui-supermerger?tab=readme-ov-file#adjust). It adds more noise and texture to the output, and can also adjust gamma/brightness using the last parameter. The 3 variants are fundamentally doing the same thing, but with different parameter controls. Scale/multiplier multiplies the S1 (weight) and S2 (bias) parameters. 

Donut Detailer: Initial try at making the node, dosent mimick supermerger accurately. 

![image](https://github.com/user-attachments/assets/b0477a38-86c2-42fd-a635-82afdef3b8a4)

Donut Detailer 2: Mimicks closes Supermerger Adjust parameters. 

![image](https://github.com/user-attachments/assets/6d0cc683-e005-481b-abe4-487700686df3)

Donut Detailer 4: Making it more barebone, without the coefficients. 

![image](https://github.com/user-attachments/assets/e1bafb1c-a24a-448e-92f9-f9e27f98157d)

Thanks to epiTune for helping me make this, and ChatGPT. Note: epiTune does not think this is the best solution to adding more texture as it is a a crude way of modifying the model, use it sparingly.
