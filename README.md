# TreeGCN-GAN
Implementation of TreeGCN-GAN paper.

# Dataset Generation Step
* ModelNet10 dataset is used.
* To sample pointcloud from mesh:
  * https://medium.com/@daviddelaiglesiacastro/3d-point-cloud-generation-from-3d-triangular-mesh-bbb602ecf238
  * https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle
  * https://www.youtube.com/watch?v=HYAgJN3x4GA
* Data generation code is present in Point_Sampling_From_Mesh folder.

# Observations / Notes
* Keeping high value for batch size helps model, in learning better real world distribution for pointcloud data.

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
* How to run the code.
* Add results.

# Reference
[1] [3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions](https://arxiv.org/abs/1905.06292) [ Dong Wook Shu, Sung Woo Park, Junseok Kwon ]
