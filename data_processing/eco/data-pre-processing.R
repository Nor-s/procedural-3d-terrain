
#' ---
#' title: "sdms_data_piping"
#' author: "Daniel Furman"
#' date: "2020"
#' ---
#'
#' This R script pipes presence and absence data for joshua trees
#'
#' Data citation:
#' GBIF.org (01 November 2020) GBIF Occurrence
#' Download https://doi.org/10.15468/dl.g6swrm
## ------------------------------------------------------------------------

library(raster)
library(rgdal)
library(dismo)
library(maptools)

name = "Ursus_arctos_Linnaeus"
name = "Theropithecus_gelada"
name = "Camel"

e <- extent(-10.82079, 25.59652, 35.96427, 65.18534) # set study area extent

e <- extent(20.96947, 55.86081, -4.26069, 22.01043)
e <- extent(-379.21145, -298.16743, 13.47514, 38.52976)
e <- extent(-16.41397,59.77011, -29.58217,  34.74464 )
## ------------------------------------------------------------------------
# pipe GBIF data
jt_raw <- read.csv(file = sprintf("data/%s_gbif.csv", name), header = TRUE, sep = "\t") # grab GBIF
# e <- extent(min(jt_raw$decimalLongitude), max(jt_raw$decimalLongitude), min(jt_raw$decimalLatitude), max(jt_raw$decimalLatitude))
jt <- data.frame(matrix(ncol = 2, nrow = length(jt_raw$decimalLongitude)))
jt[, 1] <- jt_raw$decimalLongitude
jt[, 2] <- jt_raw$decimalLatitude
jt <- unique(jt) # xantusia without duplicates
jt <- jt[complete.cases(jt), ] # remove na's
colnames(jt) <- c("lon", "lat")

# download Bioclim features
jt <- jt[which(jt$lon >= e[1] & jt$lon <= e[2]), ] # remove presences beyond extent
jt <- jt[which(jt$lat >= e[3] & jt$lat <= e[4]), ] # remove presences beyond extent
# use dismo's getData to grab climate features
bioclim.data <- getData(
  name = "worldclim",
  var = "bio",
  res = 2.5,
  path = "data/"
)
bioclim.data <- crop(bioclim.data, e * 1.25) # crop to bg point extent
plot(bioclim.data[[1]], main = "Bioclim 1")
points(jt, col = "black", pch = 16, cex = .3)

# write rasters to /data folder
dir.create(sprintf("data/%s",name), showWarnings = TRUE, recursive = FALSE, mode = "0777")

for (i in c(1:19)) {
  writeRaster(bioclim.data[[i]], paste(sprintf("data/%s/bclim",name), i, sep = ""),
    format = "ascii", overwrite = TRUE
  )
}
## ------------------------------------------------------------------------
length_presences <- length(jt$lat)
# sample background points from a slightly wider extent
bg <- randomPoints(mask = bioclim.data[[1]], n = length_presences * 2, p = jt, ext = e, extf = 1.25)

colnames(bg) <- c("lon", "lat")
train <- rbind(jt, bg) # combine with presences
pa_train <- c(rep(1, nrow(jt)), rep(0, nrow(bg))) # col of ones and zeros
train <- data.frame(cbind(CLASS = pa_train, train)) # final dataframe

# create spatial points
crs <- crs(bioclim.data[[1]])
train <- train[sample(nrow(train)), ]
class.pa <- data.frame(train)
dataMap.jt <- SpatialPointsDataFrame(train[, c(2, 3)], class.pa,
  proj4string = crs
)
# write as shp
writeOGR(dataMap.jt, sprintf("data/%s/%s.shp", name, name), name, driver = "ESRI Shapefile", overwrite_layer = TRUE)

# plot our points
plot(bioclim.data[[1]], main = "Bioclim 1")
points(bg, col = "red", pch = 16, cex = .3)
points(jt, col = "black", pch = 16, cex = .3)
