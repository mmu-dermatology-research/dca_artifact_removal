# Dermoscopic Dark Corner Artifacts Removal: Friend or Foe?

## Citation
If you use any methods, data, or code from this repository please consider citing our paper:

```BibTex
@misc{pewton2023dca,
  title  = {Dermoscopic Dark Corner Artifacts Removal: Friend or Foe?}, 
  author = {Samuel William Pewton and Bill Cassidy and Connah Kendrick and Moi Hoon Yap},
  year   = {2023},
  doi    = {10.48550/arXiv.2306.13446}
}
```

## Masks
If you only require the dark corner artifact masks from these experiments to use in your own dataset, they be downloaded from the following Kaggle Database repository:

https://www.kaggle.com/datasets/mmucomputervision/dark-corner-artifact-masks-for-isic-images


## Requirements
1. Datasets:
		- ISIC unbalanced dataset (Duplicates removed).. follow guide at https://github.com/mmu-dermatology-research/isic_duplicate_removal_strategy - save this dataset within the <code>Data</code> directory.
		- Fitzpatrick 17k.. follow guide at https://github.com/mattgroh/fitzpatrick17k - save this dataset within the <code>Data</code> directory.
		- DCA Masks.. use "Generate all DCA masks" method at https://github.com/mmu-dermatology-research/dark_corner_artifact_removal and save results within <code>Data</code> directory. <code>./Data/DCA_Masks/</code>
2. Models:
    - Download <code>EDSR.pb</code> from https://github.com/Saafke/EDSR_Tensorflow and save inside the <code>Models</code> directory. <code>/Models/EDSR_x4.pb</code>
3. Installations:
    - Python 3.9.7
		- Anaconda 4.11.0
		- pandas 1.3.5
		- numpy 1.21.5
		- scikit-learn 1.0.2
		- scikit-image 0.16.2
		- Jupyter Notebook
		- matplotlib 3.5.0
		- OpenCV 4.5.5
		- Pillow 8.4.0
		- Tensorflow 2.9.0-dev20220203
		- Tensorflow-GPU 2.9.0-dev20220203
		- CUDA 11.2.1
		- CuDNN 8.1
		- Keras

## Generating the dca split dataset
1. Open "./Modules/create_balanced_dca_dataset.py" module
2. Read through __docstring__ for module carefully - changing filepaths as necessary
3. Execute the module

## Project Steps	
1. Train the models: 		train three InceptionResNetV2 networks on each of the training/validation sets to form a model on the clean set, a model on the binary dca set, and a model on the realistic dca set. Refer to the paper for more information on the network hyper-parameters.
2. Score the models: 		score the each of the models on each of the individual test sets, this can be done with the model_performance.py module.
3. Extract the gradcam heatmaps from all images:	run the extract_gradcam.ipynb notebook. (ensure that all of the required filepaths are uncommented)
4. Calculate the brightness intensities for each of the test set images:	modify the base image filepath in the split_intensity.py module to reflect the root folder of the extracted heatmaps. Run the script to generate a .csv file for the internal and external brightness measures for each image. Once this is complete, run the calculate_intensity_averages.py module to calculate the averages across all of the images. 

**Full Model Performances on all individual testing sets:**
<table>
	<tr>
		<td>Model Used</td>
    <td>Test Set</td>
    <td colspan="3">Metrics</td>
    <td colspan="3">Micro-Average</td>
	</tr>
	<tr>
<td> </td><td> </td><td>Acc</td><td>TPR</td><td>TNR</td><td>F1</td><td>AUC</td><td>Precision</td>
</tr>
<tr><td>Clean</td><td>base-small</td><td>0.59</td><td>0.86</td><td>0.32</td><td>0.68</td><td>0.63</td><td>0.56</td></tr>
<tr><td></td><td>ns-small</td><td>0.59</td><td>0.86</td><td>0.31</td><td>0.68</td><td>0.62</td><td>0.56</td></tr>
<tr><td></td><td>telea-small</td><td>0.59</td><td>0.86</td><td>0.31</td><td>0.68</td><td>0.62</td><td>0.56</td></tr>
<tr><td> </td><td>base-medium</td><td>0.57</td><td>0.91</td><td>0.24</td><td>0.68</td><td>0.64</td><td>0.54</td></tr>
<tr><td> </td><td>ns-medium</td><td>0.62</td><td>0.88</td><td>0.36</td><td>0.70</td><td>0.68</td><td>0.58</td></tr>
<tr><td> </td><td>telea-medium</td><td>0.62</td><td>0.87</td><td>0.36</td><td>0.69</td><td>0.68</td><td>0.58</td></tr>
<tr><td> </td><td>base-large</td><td>0.51</td><td>0.99</td><td>0.01</td><td>0.67</td><td>0.58</td><td>0.50</td></tr>
<tr><td> </td><td>ns-large</td><td>0.64</td><td>0.85</td><td>0.44</td><td>0.70</td><td>0.71</td><td>0.60</td></tr>
<tr><td> </td><td>telea-large</td><td>0.65</td><td>0.85</td><td>0.45</td><td>0.71</td><td>0.71</td><td>0.61</td></tr>
<tr><td> </td><td>base-oth</td><td>0.58</td><td>0.90</td><td>0.26</td><td>0.67</td><td>0.65</td><td>0.55</td></tr>
<tr><td> </td><td>ns-oth</td><td>0.58</td><td>0.87</td><td>0.29</td><td>0.67</td><td>0.66</td><td>0.55</td></tr>
<tr><td> </td><td>telea-oth</td><td>0.58</td><td>0.87</td><td>0.29</td><td>0.67</td><td>0.66</td><td>0.55</td></tr>
<tr><td>Binary DCA</td><td>base-small</td><td>0.61</td><td>0.90</td><td>0.33</td><td>0.70</td><td>0.67</td><td>0.57</td></tr>
<tr><td></td><td>ns-small</td><td>0.61</td><td>0.89</td><td>0.33</td><td>0.70</td><td>0.67</td><td>0.57</td></tr>
<tr><td></td><td>telea-small</td><td>0.61</td><td>0.89</td><td>0.33</td><td>0.70</td><td>0.67</td><td>0.57</td></tr>
<tr><td> </td><td>base-medium</td><td>0.63</td><td>0.94</td><td>0.31</td><td>0.72</td><td>0.68</td><td>0.58</td></tr>
<tr><td> </td><td>ns-medium</td><td>0.65</td><td>0.85</td><td>0.44</td><td>0.71</td><td>0.73</td><td>0.60</td></tr>
<tr><td> </td><td>telea-medium</td><td>0.65</td><td>0.85</td><td>0.45</td><td>0.70</td><td>0.73</td><td>0.61</td></tr>
<tr><td> </td><td>base-large</td><td>0.55</td><td>0.96</td><td>0.13</td><td>0.68</td><td>0.62</td><td>0.53</td></tr>
<tr><td> </td><td>ns-large</td><td>0.70</td><td>0.79</td><td>0.61</td><td>0.73</td><td>0.75</td><td>0.67</td></tr>
<tr><td> </td><td>telea-large</td><td>0.70</td><td>0.78</td><td>0.61</td><td>0.72</td><td>0.75</td><td>0.67</td></tr>
<tr><td> </td><td>base-oth</td><td>0.60</td><td>0.83</td><td>0.36</td><td>0.67</td><td>0.67</td><td>0.57</td></tr>
<tr><td> </td><td>ns-oth</td><td>0.60</td><td>0.82</td><td>0.39</td><td>0.67</td><td>0.68</td><td>0.57</td></tr>
<tr><td> </td><td>telea-oth</td><td>0.60</td><td>0.82</td><td>0.39</td><td>0.67</td><td>0.68</td><td>0.57</td></tr>
<tr><td>Realistic DCA</td><td>base-small</td><td>0.60</td><td>0.85</td><td>0.35</td><td>0.68</td><td>0.65</td><td>0.57</td></tr>
<tr><td></td><td>ns-small</td><td>0.60</td><td>0.85</td><td>0.35</td><td>0.68</td><td>0.66</td><td>0.57</td></tr>
<tr><td></td><td>telea-small</td><td>0.60</td><td>0.84</td><td>0.36</td><td>0.68</td><td>0.66</td><td>0.57</td></tr>
<tr><td> </td><td>base-medium</td><td>0.64</td><td>0.75</td><td>0.53</td><td>0.68</td><td>0.70</td><td>0.62</td></tr>
<tr><td> </td><td>ns-medium</td><td>0.66</td><td>0.84</td><td>0.48</td><td>0.71</td><td>0.72</td><td>0.62</td></tr>
<tr><td> </td><td>telea-medium</td><td>0.66</td><td>0.82</td><td>0.49</td><td>0.71</td><td>0.73</td><td>0.62</td></tr>
<tr><td> </td><td>base-large</td><td>0.60</td><td>0.39</td><td>0.80</td><td>0.49</td><td>0.63</td><td>0.66</td></tr>
<tr><td> </td><td>ns-large</td><td>0.66</td><td>0.70</td><td>0.63</td><td>0.68</td><td>0.74</td><td>0.65</td></tr>
<tr><td> </td><td>telea-large</td><td>0.67</td><td>0.69</td><td>0.65</td><td>0.67</td><td>0.74</td><td>0.66</td></tr>
<tr><td> </td><td>base-oth</td><td>0.58</td><td>0.81</td><td>0.35</td><td>0.66</td><td>0.65</td><td>0.55</td></tr>
<tr><td> </td><td>ns-oth</td><td>0.58</td><td>0.79</td><td>0.37</td><td>0.65</td><td>0.65</td><td>0.56</td></tr>
<tr><td> </td><td>telea-oth</td><td>0.58</td><td>0.79</td><td>0.37</td><td>0.65</td><td>0.65</td><td>0.56</td></tr>
</table>

## References
```
@article{groh2021evaluating,
  title={Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset},
	author={Groh, Matthew and Harris, Caleb and Soenksen, Luis and Lau, Felix and Han, Rachel and Kim, Aerin and Koochek, Arash and Badri, Omar},
  journal={arXiv preprint arXiv:2104.09957},
  year={2021}
}

@article{cassidy2021isic,
 title   = {Analysis of the ISIC Image Datasets: Usage, Benchmarks and Recommendations},
 author  = {Bill Cassidy and Connah Kendrick and Andrzej Brodzicki and Joanna Jaworek-Korjakowska and Moi Hoon Yap},
 journal = {Medical Image Analysis},
 year    = {2021},
 issn    = {1361-8415},
 doi     = {https://doi.org/10.1016/j.media.2021.102305},
 url     = {https://www.sciencedirect.com/science/article/pii/S1361841521003509}
} 

@misc{rosebrock_2020, 
 title   = {Grad-cam: Visualize class activation maps with Keras, tensorflow, and Deep Learning}, 
 url     = {https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/}, 
 journal = {PyImageSearch}, 
 author  = {Rosebrock, Adrian}, 
 year    = {2020}, 
 month   = {3},
 note    = {[Accessed: 10-03-2022]}
} 

@article{scikit-image,
 title   = {scikit-image: image processing in {P}ython},
 author  = {van der Walt, {S}t\'efan and {S}ch\"onberger, {J}ohannes {L}. and
           {Nunez-Iglesias}, {J}uan and {B}oulogne, {F}ran\c{c}ois and {W}arner,
           {J}oshua {D}. and {Y}ager, {N}eil and {G}ouillart, {E}mmanuelle and
           {Y}u, {T}ony and the scikit-image contributors},
 year    = {2014},
 month   = {6},
 keywords = {Image processing, Reproducible research, Education,
             Visualization, Open source, Python, Scientific programming},
 volume  = {2},
 pages   = {e453},
 journal = {PeerJ},
 issn    = {2167-8359},
 url     = {https://doi.org/10.7717/peerj.453},
 doi     = {10.7717/peerj.453}
}

@article{scikit-learn,
 title   = {Scikit-learn: Machine Learning in {P}ython},
 author  = {Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal = {Journal of Machine Learning Research},
 volume  = {12},
 pages   = {2825--2830},
 year    = {2011}
}

@inproceedings{lim2017enhanced,
  title  = {Enhanced deep residual networks for single image super-resolution},
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Mu Lee, Kyoung},
  booktitle= {Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages  = {136--144},
  year   = {2017}
}
```
