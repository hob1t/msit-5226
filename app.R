#define packages to install
packages <- c('tidyverse', 'keras', 'tensorflow','reticulate')

#install all packages that are not already installed
install.packages(setdiff(packages, rownames(installed.packages())))

# if conda is not installed, uncomment this line
# reticulate::install_miniconda(update = TRUE, force = FALSE)
# install_tensorflow(extra_packages="pillow")
# install_keras()


library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)


# loading data
setwd("/Users/olegtikhonov/develop/msit-5226/indiv_project_ip")
label_list <- dir("train/")
output_n <- length(label_list)
save(label_list, file="label_list.R")

# set a size of target images
width <- 224
height<- 224
target_size <- c(width, height)
rgb <- 3 #color channels, red green blue

path_train <- "train/"
train_data_gen <- image_data_generator(rescale = 1/255, validation_split = .2) # 20% 

# train images
train_images <- flow_images_from_directory(path_train,
  train_data_gen,
  subset = 'training',
  target_size = target_size,
  class_mode = "categorical",
  shuffle=F,
  classes = label_list,
  seed = 2023)

# validate
path_valid = "valid/"
validation_images <- flow_images_from_directory(path_valid,
  train_data_gen, 
  subset = 'validation',
  target_size = target_size,
  class_mode = "categorical",
  classes = label_list,
  seed = 2023)

# show the table
show_table <- table(train_images$classes)
print("showing trained table")
print(show_table)

# show a bird
plot(as.raster(train_images[[1]][[1]][20,,,]))

# load pre-trained model
mod_base <- application_xception(weights = 'imagenet', 
   include_top = FALSE, input_shape = c(width, height, 3))
freeze_weights(mod_base) 

model_function <- function(learning_rate = 0.001, 
  dropoutrate=0.2, n_dense=1024){
  
  k_clear_session()
  
  model <- keras_model_sequential() %>%
    mod_base %>% 
    layer_global_average_pooling_2d() %>% 
    layer_dense(units = n_dense) %>%
    layer_activation("relu") %>%
    layer_dropout(dropoutrate) %>%
    layer_dense(units=output_n, activation="softmax")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(lr = learning_rate),
    metrics = "accuracy"
  )
  
  return(model)
  
}# end of function model_function


# show our model
model <- model_function()
print("Showing our trained model")
print(model)

# png("traininig_graph.jpg")


if(file.exists("bird_mod")){
  print("No train required, loading a model")
  model <- load_model_tf("bird_mod")

} else {
  print('Does not exist')
  
  # TODO: refactor to function
  batch_size <- 32
  # 6 epochs give us 65% of accuracy, let's increase to 10
  epochs <- 24
  hist <- model %>% fit(
    train_images,
    steps_per_epoch = train_images$n %/% batch_size,
    epochs = epochs,
    validation_data = validation_images,
    validation_steps = validation_images$n %/% batch_size,
    verbose = 2
  )
  
  #saving a trained model
  model %>% save_model_tf("bird_mod")
}

# testing a model
path_test <- "test/"
test_data_gen <- image_data_generator(rescale = 1/255)
test_images <- flow_images_from_directory(path_test,
   test_data_gen,
   target_size = target_size,
   class_mode = "categorical",
   classes = label_list,
   shuffle = F,
   seed = 2023)

model %>% evaluate(test_images, steps = test_images$n)


test_image <- image_load("valid/BALD EAGLE/1.jpg", target_size = target_size)
print(test_image)
x <- image_to_array(test_image)
x <- array_reshape(x, c(1, dim(x)))
x <- x/255
pred <- model %>% predict(x)
pred <- data.frame("Bird" = label_list, "Probability" = t(pred))
pred <- pred[order(pred$Probability, decreasing=T),][1:5,]
pred$Probability <- paste(format(100*pred$Probability,2),"%")
pred

print(pred)

