# http://spatialecology.weebly.com/r-code--data/category/sdm-maxent

library(dismo)  #dismo package
library(raster)
library(rgdal)
library(maptools)
require(utils)
library(rJava)  #rJava package
library("knitr")
library(rgeos)
require(jsonlite)

if (!file.exists("./data/maxent")) dir.create("./data/maxent")
if (!file.exists("./data/maxent/bioclim")) dir.create("./data/maxent/bioclim")
if (!file.exists("./data/maxent/studyarea")) dir.create("./data/maxent/studyarea")
if (!file.exists("./output")) dir.create("./output")

# utils::download.file(url = "https://raw.githubusercontent.com/mrmaxent/Maxent/master/ArchivedReleases/3.3.3k/maxent.jar", 
#     destfile = paste0(system.file("java", package = "dismo"), 
#         "/maxent.jar"), mode = "wb")  ## wb for binary file, otherwise maxent.jar can not execute


# download climate data from worldclim.org
bioclim.data <- getData(
  name = "worldclim",
  var = "bio",
  res = 2.5,
  path = "data/"
)
# crs(bioclim.data) <- CRS('+init=EPSG:4326')

# clim_list <- list.files("./data/wc2-5/", pattern = ".bil$", 
#     full.names = T) 
# clim <- raster::stack(clim_list)




if (file.exists("./data/occ_raw")) {
    load("./data/occ_raw")
} else {
    occ_raw <- gbif("Camelus")
    save(occ_raw, file = "./data/occ_raw")
    write.csv("./data/occ_raw.csv")
}

occ_clean <- subset(occ_raw, (!is.na(lat)) & (!is.na(lon)))  #  '!' means the opposite logic value
cat(nrow(occ_raw) - nrow(occ_clean), "records are removed")
dups <- duplicated(occ_clean[c("lat", "lon")])
occ_unique <- occ_clean[!dups, ]
cat(nrow(occ_clean) - nrow(occ_unique), "records are removed")

coordinates(occ_unique) <- ~lon + lat
plot(bioclim.data[[1]])  # to the first layer of the bioclim layers as a reference
plot(occ_unique, add = TRUE)  # plot the oc_unique on the above raster layer

#####Thread 9
occ_unique <- occ_unique[which(occ_unique$lon > -25 & occ_unique$lon < 
    60 & occ_unique$lat < 35), ]

#####Thread 10
# thin occ data (keep one occurrence point per cell)

cells <- cellFromXY(bioclim.data[[1]], occ_unique)
dups <- duplicated(cells)
occ_final <- occ_unique[!dups, ]
cat(nrow(occ_unique) - nrow(occ_final), "records are removed")
plot(bioclim.data[[1]])
plot(occ_final, add = T, col = "red")  # the 'add=T' tells R to put the incoming data on the existing layer

#####Thread 11
# this creates a 4-decimal-degree buffer around the
# occurrence data

occ_buff <- buffer(occ_final, 4)
plot(bioclim.data[[1]])
plot(occ_final, add = T, col = "red")  # adds occurrence data to the plot
plot(occ_buff, add = T, col = "blue")  # adds buffer polygon to the plot
print(extent(occ_buff))
#####Thread 12
studyArea <- crop(bioclim.data, extent(occ_buff)*1.25)  
# studyArea <- mask(studyArea, occ_buff)
writeRaster(studyArea,
            # a series of names for output files
            filename=paste0("./data/studyarea/",names(studyArea),".asc"), 
            format="ascii", ## the output format
            bylayer=TRUE, ## this will save a series of layers
            overwrite=T)
crs(studyArea) <- CRS('+init=EPSG:4326')

# tmpArea.data <-  crop(bioclim.data, extent(occ_buff))  

# for (i in c(1:19)) {
#   writeRaster(tmpArea.data[[i]], paste("./data/tmp/", i, sep = ""),
#     format = "ascii", overwrite = TRUE
#   )
# }
#####Thread 13
set.seed(1) 
length_presences <- length(occ_final$lat)
print(length_presences)
jt <- data.frame(matrix(ncol = 2, nrow = length(occ_final$lon)))
jt[, 1] <- occ_final$lon
jt[, 2] <- occ_final$lat
colnames(jt) <- c("lon", "lat")

bg <- sampleRandom(x=studyArea, size=length_presences*10, sp = TRUE)

plot(studyArea[[1]])
plot(bg,add=T, col = "black") 
plot(occ_final,add=T,col="red")
#####Thread 14

# randomly select 50% for training
selected <- sample(1:nrow(occ_final), nrow(occ_final) * 0.7)
occ_train <- occ_final[selected, ]  # this is the selection to be used for model training
occ_test <- occ_final[-selected, ]  # this is the opposite of the selection which will be used for model testing

#####Thread 15

p <- extract(bioclim.data, occ_train)
# env conditions for testing occ
p_test <- extract(bioclim.data, occ_test)
# extracting env conditions for background
a <- extract(bioclim.data, bg)
#####Thread 16

pa <- c(rep(1, nrow(p)), rep(0, nrow(a)))
pder <- as.data.frame(rbind(p, a))

#####Thread 17
mod <- maxent(x=pder, ## env conditions
              p=pa,   ## 1:presence or 0:absence

              path=paste0("./output/maxent_outputs"), ## folder for maxent output; 
              # if we do not specify a folder R will put the results in a temp file, 
              # and it gets messy to read those. . .
              args=c("responsecurves") ## parameter specification
              )
# mod
# mod@results
ped1 <- predict(mod, studyArea)  # studyArea is the clipped rasters 
plot(ped1)  # plot the continuous prediction

writeRaster(ped1,'test.tif', format="GTiff", overwrite=TRUE)

ped3 <- predict(mod, p)
head(ped3)
hist(ped3) 

#####Thread 19


# using 'training data' to evaluate p & a are dataframe/s
# (the p and a are the training presence and background
# points)
mod_eval_train <- dismo::evaluate(p = p, a = a, model = mod)
print(mod_eval_train)

mod_eval_test <- dismo::evaluate(p = p_test, a = a, model = mod)
print(mod_eval_test)  # training AUC may be higher than testing AUC
