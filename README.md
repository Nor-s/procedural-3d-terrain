# **Procedural3DTerrain**(+ eco)

Generate terrain with GANs by labeling environments in which species live

-   This is a fork of [this repo](https://github.com/Panagiotou/Procedural3DTerrain).

    -   Added: Conditional proGANs
    -   Changed: Input Channel(RGB to RGBA, rgb: sat, a: dem)

## **Dataset**


### **ecosystem**

for labeling

-   **GBIF**: for lat, lon

-   **Worldclim 2.5m** (about 4.5 km): for sampling

### **procedural** **3d** **terrain**

-   **Google Earth Engine**: for DEM, sat RGB Image (256x256, $13km^2$)

## **Results**

## **References**

-   SDM(species distribution model)

    -   https://github.com/daniel-furman/Python-species-distribution-modeling
    -   https://github.com/shandongfx/workshop_maxent_R/blob/master/code/Appendix1_case_study.md
    -   http://spatialecology.weebly.com/r-code--data/49
    -   https://scikit-learn.org/stable/auto_examples/applications/plot_species_distribution_modeling.html

-   GANs
    -   https://arxiv.org/abs/1411.1784
    -   https://github.com/aladdinpersson/Machine-Learning-Collection
    -   https://kr.mathworks.com/help/deeplearning/ug/train-conditional-generative-adversarial-network.html
    -   https://blog.paperspace.com/conditional-generative-adversarial-networks/
    -   https://github.com/christopher-beckham/gan-heightmaps
