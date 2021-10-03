# Tree-GAN
Implementation of Tree-GAN paper.

# Dataset Generation Step
* ModelNet10 dataset is used.
* To sample pointcloud from mesh:
  * https://medium.com/@daviddelaiglesiacastro/3d-point-cloud-generation-from-3d-triangular-mesh-bbb602ecf238
  * Area of triangle with 3 coordinates: https://math.stackexchange.com/a/1951650
  * https://www.youtube.com/watch?v=HYAgJN3x4GA
* Data generation code is present in Point_Sampling_From_Mesh folder.
* ModelNet10 numpy format dataset: [Link](https://drive.google.com/file/d/19VSP44tRHW-w9sIv8aHkXJgWqROSp5nU/view?usp=sharing)

# Observations / Notes
* Keeping larger values for batch size helps model in learning better distribution for pointcloud data.

# Pre-trained model
* Download pre-trained model from googl drive: [Link](https://drive.google.com/drive/folders/1BB39jONorejKfLeO4UQX84t3OlpSowQ0?usp=sharing)
* Keep treegan_ckpt folder as it is in code directory.

# Results
  <table style="width:100%; height:100%; border:none;">
          <tr>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65010.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65030.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65259.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65279.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65508.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65528.png" style="width:128px; height:128px;"/>
               </td>
          </tr>
          <tr>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65757.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/65777.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/66006.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/66026.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/66255.png" style="width:128px; height:128px;"/>
               </td>
               <td>
                    <img src="https://github.com/prajwalsingh/TreeGCN-GAN/blob/main/results/66275.png" style="width:128px; height:128px;"/>
               </td>
          </tr>
  </table>

# Todo Task
* <strike> Explaination of the data generation part.</strike>
* <strike> Add results.</strike>

# Reference
[1] [3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions](https://arxiv.org/abs/1905.06292) [ Dong Wook Shu, Sung Woo Park, Junseok Kwon ]
