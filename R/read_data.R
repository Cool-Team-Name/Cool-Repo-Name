library(animation)
library(data.table)

s1_labels <- fread("stage1_labels.csv", sep = ",")
s1_labels$Gen_Id <- substr(s1_labels$Id, 1, 32)

source("functions_for_reading.R")

setwd("C:/Users/wgates001/Documents/tsa/sample")
files <- list.files()
files <- files[files != "PaxHeader"]
#files <- paste0("data/", files)

ids <- substr(files, 1, 32)
sample_labels <- subset(s1_labels, Gen_Id %in% ids)


#header <- read_header(files[2])
data <- read_data(files[1])
dim(data)
image(data[,,1])


